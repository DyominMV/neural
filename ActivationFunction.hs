module ActivationFunction where

data ActivationFunction = ActivationFunction
  { eval :: Double -> Double,
    derivativeInResult :: Double -> Double
  }

logistic :: ActivationFunction
logistic =
  ActivationFunction
    (\x -> 1.0 / (1.0 + exp (- x)))
    (\y -> y * (1 - y))

th :: ActivationFunction
th =
  ActivationFunction
    tanh
    (\y -> 1 - y * y)

no :: ActivationFunction
no =
  ActivationFunction
    id
    (const 1)

periodic :: ActivationFunction
periodic =
  ActivationFunction
    sin
    (cos . asin) 