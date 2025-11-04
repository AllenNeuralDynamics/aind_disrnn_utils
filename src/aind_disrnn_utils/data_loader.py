import numpy as np
from disentangled_rnns.library import rnn_utils


def create_disrnn_dataset(
    df, ignore_policy="include", batch_size=None
) -> rnn_utils.DatasetRNN:
    """
    Creates a disrnn dataset object

    args:
    df, a trial dataframe, as created by aind_dynamic_foraging_data_utils
        must have 'ses_idx' as an column which indicates how to divide
        trials by session
    ignore_policy (str), must be "include" or "exclude", and determines
        how to use trials where the mouse did not response
    batch_size (int) input argument to disrnn dataset
    """

    # Input checking
    if "ses_idx" not in df:
        raise ValueError("df must contain index of sessions ses_idx")
    if ignore_policy not in ["include", "exclude"]:
        raise ValueError('ignore_policy must be either "include" or "exclude"')

    # Determine the number of classes in the output prediction
    if ignore_policy == "include":
        n_classes = 3
    else:
        n_classes = 2
        print("Not implemented!")

    # Format inputs
    # Make 0/1 coded reward vector
    df["rewarded"] = df["earned_reward"].astype(int)

    # Determine size of input matrix
    # Input matrix has size [# trials, # sessions, # features]
    max_session_length = df.groupby("ses_idx")["trial"].count().max() - 1
    num_sessions = len(df["ses_idx"].unique())
    num_input_features = 2
    # Pad trials to be ignored with -1
    xs = np.full((max_session_length, num_sessions, num_input_features), -1)

    # Load each session into xs
    for dex, ses_idx in enumerate(df["ses_idx"].unique()):
        temp = df.query("ses_idx == @ses_idx")
        this_xs = (
            temp[["animal_response", "rewarded"]].shift(-1).to_numpy()[:-1, :]
        )
        xs[0 : len(this_xs), dex, :] = this_xs

    # Determine size of output matrix
    # Output matrix has size [# trials, # sessions, # features]
    num_output_features = 1
    # pad trials to be ignored with -1
    ys = np.full((max_session_length, num_sessions, num_output_features), -1)

    # Load each session into ys
    for dex, ses_idx in enumerate(df["ses_idx"].unique()):
        temp = df.query("ses_idx == @ses_idx")
        this_ys = temp[["animal_response"]].to_numpy()[1:, :]
        ys[0 : len(this_ys), dex, :] = this_ys

    # Pack into a DatasetRNN object
    dataset = rnn_utils.DatasetRNN(
        ys=ys,
        xs=xs,
        y_type="categorical",
        n_classes=n_classes,
        x_names=["prev choice", "prev reward"],
        y_names=["choice"],
        batch_size=batch_size,
        batch_mode="random",
    )
    return dataset
