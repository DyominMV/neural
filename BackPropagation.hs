module BackPropagation (learnBatch) where

import ActivationFunction
    ( ActivationFunction(derivativeInResult) )
import Data.Function ((&))
import Data.List (transpose)
import Matrix ( buildMatrix, Matrix(..), transposeMatrix )
import NeuralNetwork
    ( applyLayer, Input, Layer(..), NeuralNetwork(..), Output )
import Semiring ( Semiring(plus, prod) )

data NeuronOutput = NeuronOutput {oOutput :: Double, oDerivativeValue :: Double}

type LayerOutput = [NeuronOutput]

data NeuronLearningInfo = NeuronLearningInfo {liOutput :: Double, liDelta :: Double}

type LayerLearningInfo = [NeuronLearningInfo]

computeLayerOutputs :: Layer -> Input -> LayerOutput
computeLayerOutputs layer input =
  layer
    & activators
    & map derivativeInResult
    & (const 1 :)
    & zipWith (flip ($)) output
    & zipWith NeuronOutput output
  where
    output :: Output
    output = 1 : applyLayer layer (tail input)

computeOutputsFlipped :: NeuralNetwork -> Input -> [LayerOutput]
computeOutputsFlipped network input =
  network & layers
    & foldl
      ( \outputs@(lastOutput : _) nextLayer ->
          computeLayerOutputs nextLayer (lastOutput & map oOutput) : outputs
      )
      [map (`NeuronOutput` 1) (1 : input)]

outputLayerLearningInfo :: LayerOutput -> Output -> LayerLearningInfo
outputLayerLearningInfo factOutput targetOutput =
  zipWith
    ( \(NeuronOutput outp deriv) target ->
        NeuronLearningInfo outp (- deriv * (target - outp))
    )
    factOutput
    (1 : targetOutput)

prodV :: Matrix Double -> Input -> Output
prodV m v =
  transpose [v] & Matrix
    & prod m
    & rows
    & concat

regularLayerLearningInfo :: LayerOutput -> Layer -> LayerLearningInfo -> LayerLearningInfo
regularLayerLearningInfo factOutput layer nextLayerLearningInfo =
  tail nextLayerLearningInfo
    & map liDelta
    & prodV (layer & weights & transposeMatrix)
    & zipWith (*) (factOutput & map oDerivativeValue)
    & zipWith NeuronLearningInfo (factOutput & map oOutput)

computeLearningInfo :: Output -> [Layer] -> [LayerOutput] -> [LayerLearningInfo]
computeLearningInfo targetOutput layersFlipped outputsFlipped =
  zip layersFlipped (tail outputsFlipped)
    & foldl
      ( \computedLIs@(nextLayerLearningInfo : _) (layer, factOutput) ->
          regularLayerLearningInfo factOutput layer nextLayerLearningInfo : computedLIs
      )
      [outputLayerLearningInfo (outputsFlipped & head) targetOutput]

newtype WeightDeltas = WeightDeltas {deltas :: [Matrix Double]}

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

learnBatch :: Double -> [(Input, Output)] -> NeuralNetwork -> NeuralNetwork
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
    & applyWeightDeltas network