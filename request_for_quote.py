import matplotlib.pyplot as plt
import vowpalwabbit
import random
import math
import json
import pandas as pd
from utils import hash_tuple


# For this contextual bandit problem, we define the following:

# context: dict. The context of each observation. It contains the following keys:
#     - bond: str.
#         - 2Y
#         - 3Y
#         - 5Y
#         - 10Y
#         - 30Y
#     - side: str. The side of the quote. It can be either 'bid' or 'ask'
#     - quantity: int. The quantity of the quote
#     - counterparty: str.
#         - CountrysideBroker
#         - HF-Fortress
#         - RelativeValueStrategies
#         - SleepyManager
#         - SnipperFund
#         - TankerAssetManagement
#     - mid_price: float. The mid price of the current observation
#     - competitors: int. The number of competitors in the market

# action: float. The quoted spread. mid_price + quoted_spread is the quoted price.



# Read the next_mid_dict from the json file
with open('next_mid_dict.json', 'r') as f:
    next_mid_dict = json.load(f)


def get_next_mid(context: dict) -> float:
    """Get the next mid price given the context

    Args:
        context: dict. The context of each observation.

    Returns:
        float. The next mid price given the context
    
    Raises:
        ValueError: If the next mid price is not found for the given context.
    """
    next_mid = next_mid_dict.get(hash_tuple(tuple(context.values())), -1)
    if next_mid == -1:
        raise ValueError('The next mid price is not found for the given context.')
    return next_mid


def predict_prob(context: dict, quoted_spread: float) -> float:
    """Predict the probability of the quote being accepted

    Args:
        context: dict. The context of each observation.
        quoted_spread: float. mid_price + quoted_spread is the quoted price.

    Returns:
        float. The probability of the quote being accepted
    """
    return random.uniform(0, 1)


def get_cost(
    context: dict, quoted_spread: float, min_value: float, max_value: float
) -> float:
    """Calculate the cost of the quote based on the context and the quoted spread (action)

    Args:
        context: dict. The context of each observation.
        quoted_spread: float. mid_price + quoted_spread is the quoted price.
        min_value: float. The minimum value of the quoted spread.
        max_value: float. The maximum value of the quoted spread.

    Returns:
        float. The cost given the context and the action (quoted spread).

    Raises:
        ValueError: If the side of the quote is not 'bid' or 'ask'
    """
    quoted_price = context['mid_price'] + quoted_spread
    # search for the next mid price given the current context
    next_mid_price = get_next_mid(context)
    if context['side'] == 'BID':
        return (
            (next_mid_price - quoted_price)
            * predict_prob(context, quoted_spread)
            * context['quantity']
            * -1
        )
    elif context['side'] == 'ASK':
        return (
            (quoted_price - next_mid_price)
            * predict_prob(context, quoted_spread)
            * context['quantity']
            * -1
        )
    else:
        raise ValueError('side must be either BID or ASK')


def to_vw_example_format(context, cats_label=None):
    """This function modifies (context, action, cost, probability) to VW friendly json format"""
    example_dict = {}
    if cats_label is not None:
        chosen_temp, cost, pdf_value = cats_label
        example_dict['_label_ca'] = {
            'action': chosen_temp,
            'cost': cost,
            'pdf_value': pdf_value,
        }
    example_dict['c'] = {
        f'bond={context["bond"]}': 1,
        f'side={context["side"]}': 1,
        f'quantity={context["quantity"]}': 1,
        f'counterparty={context["counterparty"]}': 1,
        f'mid_price={context["mid_price"]}': 1,
        f'competitors={context["competitors"]}': 1,
    }
    return json.dumps(example_dict)


def train_vw_model(
    vw: vowpalwabbit.Workspace,
    rfq_training_data: pd.DataFrame,
    cost_function: callable,
    min_value,
    max_value,
    do_learn=True,
):
    rewards = []
    for i, row in rfq_training_data.iterrows():
        row['Quantity'] = int(row['Notional'] / 100)
        context = {
            'bond': row['Bond'].split(' ')[-1],
            'side': row['Side'],
            'quantity': row['Quantity'],
            'counterparty': row['Counterparty'],
            'mid_price': row['MidPrice'],
            'competitors': row['Competitors'],
        }
        # get the next mid price
        quoted_spread, pdf = vw.predict(to_vw_example_format(context))

        cost = cost_function(context, quoted_spread, min_value, max_value)
        if do_learn:
            txt_ex = to_vw_example_format(context, cats_label=(quoted_spread, cost, pdf))
            vw_format = vw.parse(txt_ex, vowpalwabbit.LabelType.CONTINUOUS)
            vw.learn(vw_format)
            vw.finish_example(vw_format)
        rewards.append(-cost)
    return rewards 



if __name__ == '__main__':
    rfq_training_data = pd.read_excel('./data/rfq.xlsx', sheet_name='InSample')


    num_actions = 32
    bandwidth = 0.005

    vw = vowpalwabbit.Workspace(
    "--cats "
    + str(num_actions)
    + "  --bandwidth "
    + str(bandwidth)
    + " --min_value -0.2 --max_value 0.2 --json --chain_hash --coin --epsilon 0.2 -q :: --quiet"
)

    rewards = train_vw_model(vw, rfq_training_data, get_cost, -0.2, 0.2, do_learn=True)
    vw.finish()

    # Plot the costs
    cum_rewards = pd.Series(rewards).cumsum()
    plt.plot(rewards)
    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.title('Rewards vs Iteration')
    plt.show()