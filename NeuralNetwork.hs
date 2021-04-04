{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}

module NeuralNetwork where

import ActivationFunction (ActivationFunction, eval)
import Control.DeepSeq (NFData, force)
import Data.Function ((&))
import Data.List (transpose)
import GHC.Generics (Generic)
import Matrix (Matrix (..))
import Semiring (Semiring (prod))

type Input = [Double]

type Output = [Double]

type Batch = [([Double], [Double])]

data Layer = Layer
  { weights :: Matrix Double,
    activators :: [ActivationFunction]
  }
  deriving (NFData, Generic)

instance Show Layer where
  show layer = show (layer & weights) ++ concatMap (\x -> show x ++ " ") (layer & activators)

applyLayer :: Layer -> Input -> Output
applyLayer layer input =
  Matrix (transpose [1 : input])
    & prod (layer & weights)
    & rows
    & concat
    & zipWith (force . eval) (layer & activators)

newtype NeuralNetwork = NeuralNetwork {layers :: [Layer]} deriving (NFData, Generic)

applyNetwork :: NeuralNetwork -> Input -> Output
applyNetwork network input =
  layers network
    & foldl (force . flip applyLayer) input

networkError :: NeuralNetwork -> Input -> Output -> Double
networkError network input output =
  applyNetwork network input
    & zipWith (-) output
    & map (^ 2)
    & sum
    & force

networkBatchError :: NeuralNetwork -> Batch -> Double
networkBatchError network samples =
  samples
    & map (uncurry $ networkError network)
    & sum
    & force

instance Show NeuralNetwork where
  show nn =
    layers nn
      & concatMap (\layer -> show layer ++ "\n")

data NetworkStructure = NetworkStructure
  { numberOfInputs :: Int,
    activationFunctions :: [[ActivationFunction]]
  }
  deriving (Show, Read)
