function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

%add the leading intercept term
X = [ones(size(X,1),1) X];

%mm = number of training examples
%nn = number of features for e.g. a 20x20 image has 400 features
[mm nn] = size(X)

a1 = X;

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

%number of rows in z2 is the height of the hidden layer
%number of columns is the number of examples (since X was transposed)
%so z2 has 5000 column vectors each having the height of the hidden layer.
z2 = Theta1 * X';
a2 = sigmoid(z2);

%debug statement. remove the semi-colon to see on stdout
size(a2);

%add the bias 0th term.
a2 = [ones(1,size(a2,2));a2];
size(a2);

%number of rows is the height of the hidden layer
%number of columns is the number of training examples
z3 = Theta2 * a2;
a3 = sigmoid(z3);
size(a3);

%returns the predicted value for each input (transpose of a3) in a column vector
%p_tmp is discarded. the index vector that stores the index where the max value occurs
%is assigned to p.
[p_tmp idx] = max(a3',[],2);
size(idx);
p = idx;








% =========================================================================


end
