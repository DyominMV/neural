{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}

module BackPropagation (learnBatch) where

import ActivationFunction
  ( ActivationFunction,
    derivative,
    eval,
  )
import Control.DeepSeq (NFData, force)
import Data.Function ((&))
import Data.List (transpose)
import GHC.Generics (Generic)
import Matrix (Matrix (..), buildMatrix, transposeMatrix)
import NeuralNetwork
  ( Batch,
    Input,
    Layer (..),
    NeuralNetwork (..),
    Output,
    applyLayer,
  )
import Semiring (Semiring (plus, prod))

data NeuronOutput = NeuronOutput {oOutput :: Double, oDerivativeValue :: Double}
  deriving (NFData, Generic)

type LayerOutput = [NeuronOutput]

data NeuronLearningInfo = NeuronLearningInfo {liOutput :: Double, liDelta :: Double}
  deriving (NFData, Generic)

type LayerLearningInfo = [NeuronLearningInfo]

prodV :: Matrix Double -> Input -> Output
prodV m v =
  transpose [v] & Matrix
    & prod m
    & rows
    & concat
    & force

computeLayerOutputs :: Layer -> Input -> LayerOutput
computeLayerOutputs layer input =
  prodV (layer & weights) input
    & zipWith
      (\activator sum -> NeuronOutput (eval activator sum) (derivative activator sum))
      (layer & activators)
    & (NeuronOutput 1 1 :)
    & force

computeOutputsFlipped :: NeuralNetwork -> Input -> [LayerOutput]
computeOutputsFlipped network input =
  network & layers
    & foldl
      ( \outputs@(lastOutput : _) nextLayer ->
          computeLayerOutputs nextLayer (lastOutput & map oOutput) : outputs
      )
      [map (`NeuronOutput` 1) (1 : input)]
    & force

outputLayerLearningInfo :: LayerOutput -> Output -> LayerLearningInfo
outputLayerLearningInfo factOutput targetOutput =
  zipWith
    ( \(NeuronOutput outp deriv) target ->
        NeuronLearningInfo outp (- deriv * (target - outp))
    )
    factOutput
    (1 : targetOutput)
    & force

regularLayerLearningInfo :: LayerOutput -> Layer -> LayerLearningInfo -> LayerLearningInfo
regularLayerLearningInfo factOutput layer nextLayerLearningInfo =
  tail nextLayerLearningInfo
    & map liDelta
    & prodV (layer & weights & transposeMatrix)
    & zipWith (*) (factOutput & map oDerivativeValue)
    & zipWith NeuronLearningInfo (factOutput & map oOutput)
    & force

computeLearningInfo :: Output -> [Layer] -> [LayerOutput] -> [LayerLearningInfo]
computeLearningInfo targetOutput layersFlipped outputsFlipped =
  zip layersFlipped (tail outputsFlipped)
    & foldl
      ( \computedLIs@(nextLayerLearningInfo : _) (layer, factOutput) ->
          force $ regularLayerLearningInfo factOutput layer nextLayerLearningInfo : computedLIs
      )
      [outputLayerLearningInfo (outputsFlipped & head) targetOutput]

newtype WeightDeltas = WeightDeltas {deltas :: [Matrix Double]}
  deriving (NFData, Generic)

instance Semigroup WeightDeltas where
  (<>) wd1 wd2 =
    zipWith
      plus
      (wd1 & deltas)
      (wd2 & deltas)
      & WeightDeltas

applyWeightDeltas :: NeuralNetwork -> WeightDeltas -> NeuralNetwork
applyWeightDeltas network weightDeltas =
  zipWith
    (\(Layer ws acts) deltaMatrix -> Layer (ws `plus` deltaMatrix) acts)
    (network & layers)
    (weightDeltas & deltas)
    & NeuralNetwork
    & force

mapNeighbours :: (a -> a -> b) -> [a] -> [b]
mapNeighbours f as = zipWith f as (tail as)

computeWeightDeltas :: Double -> [LayerLearningInfo] -> WeightDeltas
computeWeightDeltas eta learningInfos =
  mapNeighbours
    ( \inputLayer ouputLayer ->
        buildMatrix (*) (ouputLayer & tail & map liDelta) (inputLayer & map liOutput)
          & fmap (* (- eta))
    )
    learningInfos
    & WeightDeltas
    & force

learnBatch :: Double -> Batch -> NeuralNetwork -> NeuralNetwork
learnBatch eta samples network =
  samples
    & map
      ( \(input, output) ->
          input
            & computeOutputsFlipped network
            & computeLearningInfo output (network & layers & reverse)
            & computeWeightDeltas eta
      )
    & foldl1 (<>)
    & force
    & applyWeightDeltas network