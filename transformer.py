import functools
from typing import Callable, Tuple, List

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

import auto_diff as ad
import torch
from torchvision import datasets, transforms

max_len = 28


def linear(input: ad.Node, weight: ad.Node, bias: ad.Node) -> ad.Node:
    return ad.matmul(input, weight)  + bias


def attention(
    input: ad.Node, W_Q: ad.Node, W_K: ad.Node, W_V: ad.Node, d: int
) -> ad.Node:
    Q = ad.matmul(input, W_Q)
    K = ad.matmul(input, W_K)
    V = ad.matmul(input, W_V)

    scaled = ad.matmul(Q, ad.transpose(K, -2, -1) / np.sqrt(d))
    A = ad.softmax(scaled)
    #compute weighted sum
    output = ad.matmul(A, V)
    return output


def feed_forward(input: ad.Node, W_O: ad.Node, b_1: ad.Node, W_1: ad.Node):
    l1 = linear(input, W_O, b_1)
    relu = ad.relu(l1)
    l2 = linear(relu, W_1, b_1)
    return l2



def transformer(
    X: ad.Node,
    nodes: List[ad.Node],
    model_dim: int,
    seq_length: int,
    eps,
    batch_size,
    num_classes,
) -> ad.Node:
    """Construct the computational graph for a single transformer layer with sequence classification.

    Parameters
    ----------
    X: ad.Node
        A node in shape (batch_size, seq_length, model_dim), denoting the input data.
    nodes: List[ad.Node]
        Nodes you would need to initialize the transformer.
    model_dim: int
        Dimension of the model (hidden size).
    seq_length: int
        Length of the input sequence.

    Returns
    -------
    output: ad.Node
        The output of the transformer layer, averaged over the sequence length for classification, in shape (batch_size, num_classes).
    """

    """TODO: Your code here"""
    W_Q, W_K, W_V, W_O, W_1, W_2, b_1, b_2 = nodes
        
    single_head_attention = attention(X, W_Q, W_K, W_V, model_dim)
    layer_norm_1 = ad.layernorm(single_head_attention, [model_dim], eps)
    
    feed_forward_output = feed_forward(layer_norm_1, W_O, b_1, W_1)
    layer_norm_2 = ad.layernorm(feed_forward_output, [model_dim], eps)
    # pooling layer to reduce the dimension
    pooling = ad.mean(layer_norm_2, dim=1, keepdim=False)
    output = ad.matmul(pooling, W_2) + b_2
    return output





def softmax_loss(Z: ad.Node, y_one_hot: ad.Node, batch_size: int) -> ad.Node:
    """Construct the computational graph of average softmax loss over
    a batch of logits.

    Parameters
    ----------
    Z: ad.Node
        A node in of shape (batch_size, num_classes), containing the
        logits for the batch of instances.

    y_one_hot: ad.Node
        A node in of shape (batch_size, num_classes), containing the
        one-hot encoding of the ground truth label for the batch of instances.

    batch_size: int
        The size of the mini-batch.

    Returns
    -------
    loss: ad.Node
        Average softmax loss over the batch.
        When evaluating, it should be a zero-rank array (i.e., shape is `()`).

    Note
    ----
    1. In this homework, you do not have to implement a numerically
    stable version of softmax loss.
    2. You may find that in other machine learning frameworks, the
    softmax loss function usually does not take the batch size as input.
    Try to think about why our softmax loss may need the batch size.
    """
    """TODO: Your code here"""
    log_softmax = ad.log(ad.softmax(Z))
    prob = ad.mul(y_one_hot, log_softmax)
    sum_class = ad.sum_op(prob, (1,))
    sum_batch = ad.sum_op(sum_class, (0,))
    output = ad.mul_by_const(sum_batch, (-1 / batch_size))
    
    return output




def sgd_epoch(
    f_run_model: Callable,
    X: torch.Tensor,
    y: torch.Tensor,
    model_weights: List[torch.Tensor],
    batch_size: int,
    lr: float,
) -> List[torch.Tensor]:
    """Run an epoch of SGD for the logistic regression model
    on training data with regard to the given mini-batch size
    and learning rate.

    Parameters
    ----------
    f_run_model: Callable
        The function to run the forward and backward computation
        at the same time for logistic regression model.
        It takes the training data, training label, model weight
        and bias as inputs, and returns the logits, loss value,
        weight gradient and bias gradient in order.
        Please check `f_run_model` in the `train_model` function below.

    X: torch.Tensor
        The training data in shape (num_examples, in_features).

    y: torch.Tensor
        The training labels in shape (num_examples,).

    model_weights: List[torch.Tensor]
        The model weights in the model.

    batch_size: int
        The mini-batch size.

    lr: float
        The learning rate.

    Returns
    -------
    model_weights: List[torch.Tensor]
        The model weights after update in this epoch.

    b_updated: torch.Tensor
        The model weight after update in this epoch.

    loss: torch.Tensor
        The average training loss of this epoch.
    """

    """TODO: Your code here"""
    num_examples = X.shape[0]
    num_batches = (num_examples + batch_size - 1) // batch_size  # Compute the number of batches
    total_loss = 0.0
    
    for i in range(num_batches):
        # Get the mini-batch data
        start_idx = i * batch_size
        if start_idx + batch_size> num_examples:continue
        end_idx = min(start_idx + batch_size, num_examples)
        X_batch = X[start_idx:end_idx, :max_len]
        y_batch = y[start_idx:end_idx]

        # Compute forward and backward passes
        # TODO: Your code here
 
        W_Q_sgd, W_K_sgd, W_V_sgd, W_O_sgd, W_1_sgd, W_2_sgd, b_1_sgd, b_2_sgd = (
            model_weights[0],
            model_weights[1],
            model_weights[2],
            model_weights[3],
            model_weights[4],
            model_weights[5],
            model_weights[6],
            model_weights[7],
        )
        
  

        # Update the model weights list        
        res = f_run_model(
            {
                "X_batch": X_batch,  
                "y_batch": y_batch,  
                "W_Q": W_Q_sgd,
                "W_K": W_K_sgd,
                "W_V": W_V_sgd,
                "W_O": W_O_sgd,
                "W_1": W_1_sgd,
                "W_2": W_2_sgd,
                "b_1": b_1_sgd,
                "b_2": b_2_sgd,
            }
        )
        # Update weights and biases
        # TODO: Your code here
        # Hint: You can update the tensor using something like below:
        # W_Q -= lr * grad_W_Q.sum(dim=0)
        (
            logits,
            loss,
            grad_W_Q,
            grad_W_K,
            grad_W_V,
            grad_W_O,
            grad_W_1,
            grad_W_2,
            grad_b_1,
            grad_b_2,
        ) = res

        W_Q_sgd -= lr * grad_W_Q.sum(dim=0)
        W_K_sgd -= lr * grad_W_K.sum(dim=0)
        W_V_sgd -= lr * grad_W_V.sum(dim=0)
        W_O_sgd -= lr * grad_W_O.sum(dim=0)
        W_1_sgd -= lr * grad_W_1.sum(dim=0)
        W_2_sgd -= lr * grad_W_2.sum(dim=0)
        b_1_sgd -= lr * grad_b_1.sum(dim=(0, 1))
        b_2_sgd -= lr * grad_b_2.sum(dim=(0, 1))

        # Accumulate the loss
        # TODO: Your code here
        total_loss += loss.sum(dim=0)

    # Compute the average loss

    average_loss = total_loss / num_examples
    print("Avg_loss:", average_loss)

    # TODO: Your code here
    # You should return the list of parameters and the loss
    model_weights = [W_Q_sgd, W_K_sgd, W_V_sgd, W_O_sgd, W_1_sgd, W_2_sgd, b_1_sgd, b_2_sgd]
    return model_weights, average_loss




def train_model():
    """Train a logistic regression model with handwritten digit dataset.

    Note
    ----
    Your implementation should NOT make changes to this function.
    """
    # Set up model params

    # TODO: Tune your hyperparameters here
    # Hyperparameters
    input_dim = 28  # Each row of the MNIST image
    seq_length = max_len  # Number of rows in the MNIST image
    num_classes = 10  #
    model_dim = 128  #
    eps = 1e-5
    #seq_len 28
    # - Set up the training settings.
    num_epochs = 20
    batch_size = 50
    lr = 0.02

    # TODO: Define the forward graph.
    X_node = ad.Variable("X")
    
    W_Q_t = ad.Variable("W_Q")
    W_K_t = ad.Variable("W_K")
    W_V_t = ad.Variable("W_V")
    W_O_t = ad.Variable("W_O")
    W_1_t = ad.Variable("W_1")
    W_2_t = ad.Variable("W_2")
    b_1_t = ad.Variable("b_1")
    b_2_t = ad.Variable("b_2")
    
    forward_nodes = [W_Q_t, W_K_t, W_V_t, W_O_t, W_1_t, W_2_t, b_1_t, b_2_t]
    
    
    y_predict: ad.Node = transformer(
        X_node, forward_nodes, model_dim, seq_length, eps, batch_size, num_classes
    )  # TODO: The output of the forward pass
    y_groundtruth = ad.Variable(name="y")

    loss: ad.Node = softmax_loss(y_predict, y_groundtruth, batch_size)
    
    
    # TODO: Construct the backward graph.
    

    # TODO: Create the evaluator.
    grads: List[ad.Node] = ad.gradients(
        loss, forward_nodes
    )  # TODO: Define the gradient nodes here
    evaluator = ad.Evaluator([y_predict, loss, *grads])
    test_evaluator = ad.Evaluator([y_predict])
    # - Load the dataset.
    #   Take 80% of data for training, and 20% for testing.
    # Prepare the MNIST dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, transform=transform, download=True
    )


    # Convert the train dataset to NumPy arrays
    X_train = (
        train_dataset.data.numpy().reshape(-1, 28, 28) / 255.0
    )  # Flatten to 784 features
    y_train = train_dataset.targets.numpy()

    # Convert the test dataset to NumPy arrays
    X_test = (
        test_dataset.data.numpy().reshape(-1, 28, 28) / 255.0
    )  # Flatten to 784 features
    y_test = test_dataset.targets.numpy()

    # Initialize the OneHotEncoder
    encoder = OneHotEncoder(
        sparse_output=False
    )  # Use sparse=False to get a dense array

    # Fit and transform y_train, and transform y_test
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))

    num_classes = 10

    # Initialize model weights.
    np.random.seed(0)
    stdv = 1.0 / np.sqrt(num_classes)
    W_Q_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_K_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_V_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_O_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_1_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_2_val = np.random.uniform(-stdv, stdv, (model_dim, num_classes))
    b_1_val = np.random.uniform(-stdv, stdv, (model_dim,))
    b_2_val = np.random.uniform(-stdv, stdv, (num_classes,))



    def f_run_model(model_weights):
        """The function to compute the forward and backward graph.
        It returns the logits, loss, and gradients for model weights.
        """
        result = evaluator.run(
            input_values={
                # TODO: Fill in the mapping from variable to tensor
                X_node: model_weights["X_batch"],
                y_groundtruth: model_weights["y_batch"],
                forward_nodes[0]: model_weights["W_Q"],
                forward_nodes[1]: model_weights["W_K"],
                forward_nodes[2]: model_weights["W_V"],
                forward_nodes[3]: model_weights["W_O"],
                forward_nodes[4]: model_weights["W_1"],
                forward_nodes[5]: model_weights["W_2"],
                forward_nodes[6]: model_weights["b_1"],
                forward_nodes[7]: model_weights["b_2"],
         
            }
        )

        return result

    def f_eval_model(X_val, model_weights: List[torch.Tensor]):
        """The function to compute the forward graph only and returns the prediction."""
        num_examples = X_val.shape[0]
        num_batches = (
            num_examples + batch_size - 1
        ) // batch_size  # Compute the number of batches
        total_loss = 0.0
        all_logits = []
        for i in range(num_batches):
            # Get the mini-batch data
            start_idx = i * batch_size
            if start_idx + batch_size > num_examples:
                continue
            end_idx = min(start_idx + batch_size, num_examples)
            X_batch = X_val[start_idx:end_idx, :max_len]
            
            
            logits = test_evaluator.run(
                {
                    # TODO: Fill in the mapping from variable to tensor
                    X_node: X_batch,
                    W_Q_t: model_weights[0],
                    W_K_t: model_weights[1],
                    W_V_t: model_weights[2],
                    W_O_t: model_weights[3],
                    W_1_t: model_weights[4],
                    W_2_t: model_weights[5],
                    b_1_t: model_weights[6],
                    b_2_t: model_weights[7],
                }
            )
            all_logits.append(logits[0])
        # Concatenate all logits and return the predicted classes
        concatenated_logits = np.concatenate(all_logits, axis=0)
        predictions = np.argmax(concatenated_logits, axis=1)
        return predictions


    # Train the model.
    X_train, X_test, y_train, y_test = (
        torch.tensor(X_train),
        torch.tensor(X_test),
        torch.DoubleTensor(y_train),
        torch.DoubleTensor(y_test),
    )
    # TODO: Initialize the model weights here
    
    
    W_Q_tensor = torch.tensor(W_Q_val)
    W_K_tensor = torch.tensor(W_K_val)
    W_V_tensor = torch.tensor(W_V_val)
    W_O_tensor = torch.tensor(W_O_val)
    W_1_tensor = torch.tensor(W_1_val)
    W_2_tensor = torch.tensor(W_2_val)
    b_1_tensor = torch.tensor(b_1_val)
    b_2_tensor = torch.tensor(b_2_val)
    
    model_weights: List[torch.Tensor] = [
        W_Q_tensor, W_K_tensor, W_V_tensor, W_O_tensor, W_1_tensor, W_2_tensor, b_1_tensor, b_2_tensor
    ]

    for epoch in range(num_epochs):
        X_train, y_train = shuffle(X_train, y_train)
        model_weights, loss_val = sgd_epoch(
            f_run_model, X_train, y_train, model_weights, batch_size, lr
        )
        # Evaluate the model on the test data.
        predict_label = f_eval_model(X_test, model_weights)
        print(
            f"Epoch {epoch}: test accuracy = {np.mean(predict_label== y_test.numpy())}, "
            f"loss = {loss_val}"
        )
    # Return the final test accuracy.
    predict_label = f_eval_model(X_test, model_weights)
    return np.mean(predict_label == y_test.numpy())


if __name__ == "__main__":
    print(f"Final test accuracy: {train_model()}")
