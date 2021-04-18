module Main where

import ActivationFunction (ActivationFunction (..))
import BackPropagation (learnBatch)
import Control.DeepSeq (force)
import Data.Function ((&))
import GHC.Conc (par, pseq)
import GHC.List (foldl1')
import NeuralNetwork
  ( Batch,
    NetworkStructure (NetworkStructure),
    NeuralNetwork,
    applyNetwork,
    networkBatchError,
  )
import Random (getRandomDoubles)
import RandomNetwork (getRandomNetwork, getURandomNetwork)
import System.IO
  ( IOMode (AppendMode),
    hClose,
    hPutStrLn,
    openFile,
  )
import Matrix (buildMatrix, Matrix (rows))

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

epoch :: Double -> [Batch] -> NeuralNetwork -> NeuralNetwork
epoch eta batches network =
  batches
    & foldl (flip (learnBatch eta)) network

handleNetwork :: FilePath -> Int -> [Batch] -> Batch -> Double -> NetworkStructure -> IO ()
handleNetwork filePath epochCount trainingBatches testBatch eta networkStructure = do
  neuralNetwork <- getURandomNetwork 0.1 networkStructure
  echo $ "First state of network:\n" ++ show neuralNetwork
  iterate (epoch eta trainingBatches) neuralNetwork
    & take epochCount
    & zip [1 ..]
    & mapBodyAndTail
      echoStepError
      ( \p@(step, nw) -> do
          echoStepError p
          echo ("\nLast state of network:\n" ++ show nw ++ "\n\n")
          buildMatrix (\y x -> applyNetwork nw [x, y]) [-16, 0.5 - 16 .. 16] [-16, 0.5 - 16 .. 16]
           & fmap (\[a] -> a)
           & rows
           & concatMap ((++ "; ") . concatMap (\z -> show z ++ " "))
           & (\a -> '[' : a ++ "]")
           & echo
      )
    & sequence_
  where
    echo str = do
      fileHandle <- openFile filePath AppendMode
      hPutStrLn fileHandle str
      hClose fileHandle
    echoStepError (step, nw) =
      echo
        ( show step
            ++ ": learn error = "
            ++ show (nw `networkBatchError` concat trainingBatches)
            ++ ": test error = "
            ++ show (nw `networkBatchError` testBatch)
        )
    mapBodyAndTail _ _ [] = []
    mapBodyAndTail _ tailMapper [x] = [tailMapper x]
    mapBodyAndTail bodyMapper tailMapper (x : xs) =
      let resX = bodyMapper x
       in resX : mapBodyAndTail bodyMapper tailMapper xs

targetFunction :: Double -> Double -> Double
targetFunction x y = (sin (x - y) - 2 * sin x) * 2 + y

simpleStructure :: [Int] -> NetworkStructure
simpleStructure layers =
  NetworkStructure
    2
    ( map (`replicate` Logistic) layers ++ [[No]]
    )

testData :: Batch
testData =
  [ ([x, y], [targetFunction x y])
    | x <- [-16, 0.125 - 16 .. 16],
      y <- [-16, 0.125 - 16 .. 16]
  ]

sampleBatch :: Double -> [Batch]
sampleBatch step =
  [ ([x, y], [targetFunction x y])
    | x <- [-16, step - 16 .. 16],
      y <- [-16, step - 16 .. 16]
  ]
    & chunksOf 10

randomBatch :: Int -> IO [Batch]
randomBatch samplesCount = do
  randoms <- getRandomDoubles 16
  randoms
    & chunksOf 2
    & take samplesCount
    & map (\[x, y] -> ([x, y], [targetFunction x y]))
    & chunksOf 10
    & return

chunksOf :: Int -> [a] -> [[a]]
chunksOf _ [] = []
chunksOf count list = take count list : chunksOf count (drop count list)

-- experiments

-- Обучающая выборка с шагом, уменьшающимся в 2 раза (от 2 до 0.125) (+ случайная выборка той же мощности)
-- 2 слоя по 10 нейронов, 300 эпох
experiment1 :: IO ()
experiment1 = do
  let steps = 2 : 1 : ([1 .. 3] & map (0.5 ^))
  steps
    & map
      ( \step -> do
          handleNetwork ("exp1_regular_" ++ show step) 300 (sampleBatch step) testData 0.0001 (simpleStructure [10, 10])
          let count = (32.0 / step) ^ 2 & round
          randomB <- randomBatch count
          handleNetwork ("exp1_random_" ++ show count) 300 randomB testData 0.0001 (simpleStructure [10, 10])
      )
    & foldl1 (>>)

-- 700 эпох
-- 2 слоя по 10 нейронов, шаг - 0.25
experiment2 :: IO ()
experiment2 = do
  handleNetwork "exp2" 3000 (sampleBatch 0.25) testData 0.0001 (simpleStructure [10, 10])

-- Конфигурации сети (слои от 1 до 4) (нейроны на слое: 10, 15, 20, 25, 30)
-- 300 эпох, шаг 0.25
experiment3 :: IO ()
experiment3 = do
  let namedStructure = (\l n -> (show l ++ "l " ++ show n ++ "n", simpleStructure (replicate l n)))
  let structures = [namedStructure layers neurons | layers <- [1 .. 4], neurons <- [10, 15 .. 30]]
  structures
    & map
      ( \(name, structure) -> do
          handleNetwork ("exp3_" ++ name) 300 (sampleBatch 0.25) testData 0.0001 structure
      )
    & foldl1 (>>)

main :: IO ()
main = do
  -- experiment1
  experiment2
  -- experiment3