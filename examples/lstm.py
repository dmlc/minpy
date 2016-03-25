import minpy.numpy as np
import minpy.numpy.random as npr

def sigmoid(x):
    return 0.5*(np.tanh(x / 2.0) + 1.0)   # Output ranges from 0 to 1.

def activations(weights, *args):
    cat_state = np.concatenate(args + (np.ones((args[0].shape[0],1)),), axis=1)
    return np.dot(cat_state, weights)

def logsumexp(X, axis=1):
    max_X = np.max(X)
    return max_X + np.log(np.sum(np.exp(X - max_X), axis=axis, keepdims=True))

def build_lstm(input_size, state_size, output_size):
    """Builds functions to compute the output of an LSTM."""
    weights = minpy.WeightGroup()
    weights.add_shape('init_cells',   (1, state_size))
    weights.add_shape('init_hiddens', (1, state_size))
    weights.add_shape('change',  (input_size + state_size + 1, state_size))
    weights.add_shape('forget',  (input_size + 2 * state_size + 1, state_size))
    weights.add_shape('ingate',  (input_size + 2 * state_size + 1, state_size))
    weights.add_shape('outgate', (input_size + 2 * state_size + 1, state_size))
    weights.add_shape('predict', (state_size + 1, output_size))

    def update_lstm(input, hiddens, cells, weights):
        """One iteration of an LSTM layer."""
        change  = np.tanh(activations(weights.change, input, hiddens))
        forget  = sigmoid(activations(weights.forget, input, cells, hiddens))
        ingate  = sigmoid(activations(weights.ingate, input, cells, hiddens))
        cells   = cells * forget + ingate * change
        outgate = sigmoid(activations(weights.outgate, input, cells, hiddens))
        hiddens = outgate * np.tanh(cells)
        return hiddens, cells

    def hiddens_to_output_probs(predict_weights, hiddens):
        output = activations(predict_weights, hiddens)
        return output - logsumexp(output)     # Normalize log-probs.

    def outputs(inputs, forget_weights, change_weights, ingate_weights,
            outgate_weights, predict_weights):
        """Outputs normalized log-probabilities of each character, plus an
           extra one at the end."""
        num_sequences = inputs.shape[1]
        hiddens = np.repeat(parser.get(weights, 'init_hiddens'), num_sequences, axis=0)
        cells   = np.repeat(parser.get(weights, 'init_cells'),   num_sequences, axis=0)

        output = [hiddens_to_output_probs(predict_weights, hiddens)]
        for input in inputs:  # Iterate over time steps.
            hiddens, cells = update_lstm(input, hiddens, cells, forget_weights,
                                         change_weights, ingate_weights, outgate_weights)
            output.append(hiddens_to_output_probs(predict_weights, hiddens))
        return output

    def log_likelihood(weights, inputs, targets):
        logprobs = outputs(weights, inputs)
        loglik = 0.0
        num_time_steps, num_examples, _ = inputs.shape
        for t in range(num_time_steps):
            loglik += np.sum(logprobs[t] * targets[t])
        return loglik / (num_time_steps * num_examples)

    return outputs, log_likelihood, parser.num_weights

def string_to_one_hot(string, maxchar):
    """Converts an ASCII string to a one-of-k encoding."""
    ascii = np.array([ord(c) for c in string]).T
    return np.array(ascii[:,None] == np.arange(maxchar)[None, :], dtype=int)

def one_hot_to_string(one_hot_matrix):
    return "".join([chr(np.argmax(c)) for c in one_hot_matrix])

def build_dataset(filename, sequence_length, alphabet_size, max_lines=-1):
    """Loads a text file, and turns each line into an encoded sequence."""
    with open(filename) as f:
        content = f.readlines()
    content = content[:max_lines]
    content = [line for line in content if len(line) > 2]   # Remove blank lines
    seqs = np.zeros((sequence_length, len(content), alphabet_size))
    for ix, line in enumerate(content):
        padded_line = (line + " " * sequence_length)[:sequence_length]
        seqs[:, ix, :] = string_to_one_hot(padded_line, alphabet_size)
    return seqs

if __name__ == '__main__':
    npr.seed(1)
    input_size = output_size = 128   # The first 128 ASCII characters are the common ones.
    state_size = 40
    seq_length = 30
    param_scale = 0.01
    train_iters = 100

    train_inputs = build_dataset(__file__, seq_length, input_size, max_lines=60)

    pred_fun, loglike_fun, num_weights = build_lstm(input_size, state_size, output_size)

    def print_training_prediction(weights):
        print("Training text                         Predicted text")
        logprobs = np.asarray(pred_fun(weights, train_inputs))
        for t in range(logprobs.shape[1]):
            training_text  = one_hot_to_string(train_inputs[:,t,:])
            predicted_text = one_hot_to_string(logprobs[:,t,:])
            print(training_text.replace('\n', ' ') + "|" + predicted_text.replace('\n', ' '))

    # Wrap function to only have one argument, for scipy.minimize.
    def training_loss(weights):
        return -loglike_fun(weights, train_inputs, train_inputs)

    def callback(weights):
        print("Train loss:", training_loss(weights))
        print_training_prediction(weights)

   # Build gradient of loss function using autograd.
    training_loss_and_grad = value_and_grad(training_loss)

    init_weights = npr.randn(num_weights) * param_scale

    print("Training LSTM...")
    result = minimize(training_loss_and_grad, init_weights, jac=True, method='CG',
                      options={'maxiter':train_iters}, callback=callback)
    trained_weights = result.x
