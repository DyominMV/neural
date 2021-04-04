module Main where

import ActivationFunction (ActivationFunction (..))
import BackPropagation (learnBatch)
import Data.Function ((&))
import NeuralNetwork
  ( Batch,
    NetworkStructure (NetworkStructure),
    NeuralNetwork,
    networkBatchError,
  )
import Random (getRandomDoubles)
import RandomNetwork (getRandomNetwork)
import System.IO
  ( IOMode (AppendMode),
    hClose,
    hPutStrLn,
    openFile,
  )
import GHC.Conc (par, pseq)
import GHC.List (foldl1')
import Control.DeepSeq (force)

combinationsWithRepetitions :: Int -> [a] -> [[a]]
combinationsWithRepetitions maxNumber xs =
  combsBySize xs
    & tail
    & take maxNumber
    & concat
  where
    combsBySize = foldr f ([[]] : repeat [])
    f x = scanl1 $ (++) . map (x :)

allStructures :: Int -> Int -> [ActivationFunction] -> [NetworkStructure]
allStructures maxLayers maxNeurons activators =
  iterate nPlusOneLayersVariants singleLayerVariants
    & take maxLayers
    & concat
    & map (NetworkStructure 2 . (++ [[No]]))
  where
    possibleLayers :: [[ActivationFunction]]
    possibleLayers = combinationsWithRepetitions maxNeurons activators
    singleLayerVariants :: [[[ActivationFunction]]]
    singleLayerVariants = map (: []) possibleLayers
    nPlusOneLayersVariants :: [[[ActivationFunction]]] -> [[[ActivationFunction]]]
    nPlusOneLayersVariants prevVariants =
      [ newLayer : prevVariant
        | newLayer <- possibleLayers,
          prevVariant <- prevVariants
      ]

activatorName :: ActivationFunction -> Char
activatorName activator = case activator of
  Periodic -> 'p'
  Logistic -> 'l'
  Th -> 't'
  No -> 'n'
  Gauss -> 'g'

structureName :: NetworkStructure -> String
structureName (NetworkStructure _ activators) = concatMap (('_' :) . map activatorName) activators

-- Here starts main code

latticeSamples :: Batch
latticeSamples =
  let coord = [-16, -16 + 0.25 .. 16]
   in [([x, y], [targetFunction x y]) | x <- coord, y <- coord]

monteCarloSamples :: IO Batch
monteCarloSamples = do
  randomDoubles <- getRandomDoubles 16
  let generateSamples =
        ( \(x : y : coords) ->
            ([x, y], [targetFunction x y]) : generateSamples coords
        )
  return (randomDoubles & generateSamples & take 10000)

epoch :: Batch -> NeuralNetwork -> NeuralNetwork
epoch batch network =
  batch
    & foldl (\nw b -> learnBatch eta [b] nw) network

handleNetwork :: FilePath -> Batch -> Batch -> NeuralNetwork -> IO ()
handleNetwork filePath trainingBatch testBatch network = do
  echo $ "First state of network:\n" ++ show network
  iterate (epoch trainingBatch) network
    & take epochCount
    & zip [1 ..]
    & mapBodyAndTail
      echoStepError
      ( \p@(step, nw) -> do
          echoStepError p
          echo ("\nLast state of network:\n" ++ show nw)
      )
    & sequence_
  where
    mapBodyAndTail _ _ [] = []
    mapBodyAndTail _ tailMapper [x] = [tailMapper x]
    mapBodyAndTail bodyMapper tailMapper (x : xs) =
      let resX = bodyMapper x
       in resX : mapBodyAndTail bodyMapper tailMapper xs
    echo str = do
      fileHandle <- openFile filePath AppendMode
      hPutStrLn fileHandle str
      hClose fileHandle
    echoStepError (step, nw) =
      echo
        ( show step
            ++ ": learn error = "
            ++ show (nw `networkBatchError` trainingBatch)
            ++ ": test error = "
            ++ show (nw `networkBatchError` testBatch)
        )

handleStructure :: Int -> Batch -> Batch -> NetworkStructure -> IO ()
handleStructure times latticeBatch randomBatch networkStructure =
  [ do
      let fileName = (networkStructure & structureName) ++ show x
      rn <- getRandomNetwork 0.1 networkStructure
      handleNetwork ('L' : fileName) latticeBatch latticeBatch rn
      handleNetwork ('M' : fileName) randomBatch latticeBatch rn
      putStrLn ((networkStructure & structureName) ++ show x)
    | x <- [1 .. times]
  ]
    & sequence_

targetFunction :: Double -> Double -> Double
targetFunction x y = (sin (x - y) - 2 * sin x) * 2 + y

eta :: Double
eta = 0.001

epochCount :: Int
epochCount = 100

main :: IO ()
main = do
  randomBatch <- monteCarloSamples
  let latticeBatch = latticeSamples
  map (handleStructure 3 latticeBatch randomBatch) (allStructures 3 4 [Logistic, Periodic])
    & foldl1 (>>)
