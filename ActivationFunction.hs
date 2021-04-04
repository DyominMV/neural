{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
module ActivationFunction where

import Control.DeepSeq (NFData)
import GHC.Generics (Generic)
data ActivationFunction = Logistic | Th | No | Periodic | Gauss
  deriving (Show, Read, NFData, Generic)

eval :: ActivationFunction -> Double -> Double
eval activator = case activator of
  Logistic -> \x -> 1.0 / (1.0 + exp (- x))
  Th -> tanh
  No -> id
  Periodic -> sin
  Gauss -> \x -> exp (- x * x)

derivative :: ActivationFunction -> Double -> Double
derivative activator = case activator of
  Logistic -> \x -> eval Logistic x * (1 - eval Logistic x)
  Th -> \x -> 1 - tanh x * tanh x
  No -> const 1
  Periodic -> cos
  Gauss -> \x -> - exp (- x * x) * 2 * x