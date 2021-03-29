module ActivationFunction where

data ActivationFunction = ActivationFunction
  { eval :: Double -> Double,
    derivative :: Double -> Double
  }

logistic :: ActivationFunction
logistic =
  ActivationFunction
    sigma
    (\y -> sigma y * (1 - sigma y))
  where
    sigma x = 1.0 / (1.0 + exp (- x))

th :: ActivationFunction
th =
  ActivationFunction
    tanh
    (\y -> 1 - tanh y * tanh y)

no :: ActivationFunction
no =
  ActivationFunction
    id
    (const 1)

periodic :: ActivationFunction
periodic =
  ActivationFunction
    sin
    cos

gauss :: ActivationFunction
gauss = 
  ActivationFunction
    (\x -> exp (-x*x))
    (\y -> - exp (-y*y) * 2 * y)