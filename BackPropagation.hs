{-# LANGUAGE TupleSections #-}

module BackPropagation (learnBatch) where

import ActivationFunction (derivative, eval)
import Control.DeepSeq (force)
import Data.Function ((&))
import Matrix
  ( Matrix (Matrix),
    buildMatrix,
    cols,
    transposeMatrix,
  )
import NeuralNetwork
  ( Batch,
    Input,
    Layer (..),
    NeuralNetwork (..),
    Output,
  )
import Semiring (Semiring (plus, prod))

type OutputAndDerivative = ([(Double, Double)], Maybe Layer)

type OutputAndDeltas = ([(Double, Double)], Maybe Layer)

vector :: [a] -> Matrix a
vector as = Matrix $ map (: []) as

unvector :: Matrix a -> [a]
unvector m = head $ cols m

foldToList :: (a -> b -> a) -> a -> [b] -> [a]
foldToList f a [] = [a]
foldToList f a (b : bs) = a : foldToList f (f a b) bs

outputsAndDerivatives :: Input -> NeuralNetwork -> [OutputAndDerivative]
outputsAndDerivatives inp (NeuralNetwork layers) =
  foldToList
    ( \outputAndDerivative layer ->
        outputAndDerivative
          & map fst
          & vector
          & prod (layer & weights)
          & unvector
          & zipWith
            (\activator sum -> (eval activator sum, derivative activator sum))
            (layer & activators)
          & ((1, 1) :)
    )
    (map (,1) $ 1 : inp)
    layers
    & flip zip (Nothing : map Just layers)

reversedOutputsAndDeltas :: Output -> [OutputAndDerivative] -> [OutputAndDeltas]
reversedOutputsAndDeltas targetOutput reverseOutputs =
  foldToList
    ( \(outpWithDeltas, Just layer) (outpWithDerivatives, mbLayer) ->
        outpWithDeltas
          & tail
          & map snd
          & vector
          & prod (layer & weights & transposeMatrix)
          & unvector
          & zipWith
            (\(output, deriv) sumDelta -> (output, 2 * deriv * sumDelta))
            outpWithDerivatives
          & (,mbLayer)
    )
    ( fst (head reverseOutputs)
        & zipWith
          (\t (out, der) -> (out,  - 2 * der * (t - out)))
          (1 : targetOutput)
        & (,snd $ head reverseOutputs)
    )
    (tail reverseOutputs)

mapNeighbours :: (a -> a -> b) -> [a] -> [b]
mapNeighbours f [] = []
mapNeighbours f [a] = []
mapNeighbours f (a1 : a2 : as) = f a1 a2 : mapNeighbours f (a2 : as)

computeWeightDeltas :: [OutputAndDeltas] -> [Matrix Double]
computeWeightDeltas =
  mapNeighbours
    (\(leftOutputs, _) (rightOutputs, _) -> buildMatrix (*) (tail $ map snd rightOutputs) (map fst leftOutputs))

learnBatch :: Double -> Batch -> NeuralNetwork -> NeuralNetwork
learnBatch eta batch network =
  batch
    & map
      ( \(inp, outp) ->
          outputsAndDerivatives inp network
            & reverse
            & reversedOutputsAndDeltas outp
            & reverse
            & computeWeightDeltas
      )
    & foldl1 (zipWith plus)
    & zipWith
      ( \(Layer ws as) matrix ->
          Layer (ws `plus` fmap (\x -> - x * eta / fromIntegral (length batch)) matrix) as
      )
      (network & layers)
    & NeuralNetwork
    & force