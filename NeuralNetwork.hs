module NeuralNetwork where

import ActivationFunction ( ActivationFunction(eval) )
import Data.Function ((&))
import Data.List (transpose)
import Matrix ( Matrix(..) )
import Semiring ( Semiring(prod) )

data Layer = Layer
  { weights :: Matrix Double,
    activators :: [ActivationFunction]
  }

applyLayer :: Layer -> Input -> Output
applyLayer layer input =
  Matrix (transpose [1 : input])
    & prod (layer & weights)
    & rows
    & concat
    & zipWith eval (layer & activators)

newtype NeuralNetwork = NeuralNetwork {layers :: [Layer]}

applyNetwork :: NeuralNetwork -> Input -> Output
applyNetwork network input =
  layers network
    & foldl (flip applyLayer) input

networkError :: NeuralNetwork -> Input -> Output -> Double
networkError network input output =
  applyNetwork network input
    & zipWith (-) output
    & map (^ 2)
    & sum

networkBatchError :: NeuralNetwork -> [(Input, Output)] -> Double
networkBatchError network samples =
  samples
    & map (uncurry $ networkError network)
    & sum

instance Show NeuralNetwork where
  show nn =
    layers nn
      & zip [1, 2 ..]
      & concatMap (\(num, layer) -> "Layer " ++ show num ++ "\n" ++ show (layer & weights))

type Input = [Double]

type Output = [Double]