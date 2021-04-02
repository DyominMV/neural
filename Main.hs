module Main where

import ActivationFunction ( ActivationFunction(..) )
import BackPropagation ( learnBatch )
import Data.Function ((&))
import NeuralNetwork
    ( networkBatchError,
      Batch,
      NetworkStructure(NetworkStructure),
      NeuralNetwork )
import Random ( getRandomDoubles )
import RandomNetwork ( getRandomNetwork )
import System.IO
    ( hClose, openFile, hPutStrLn, IOMode(AppendMode) )

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

handleLattice :: FilePath -> Batch -> NeuralNetwork -> IO ()
handleLattice filePath latticeBatch network = do
  fileHandle <- openFile filePath AppendMode
  let echo = hPutStrLn fileHandle
  echo "First state of network:"
  echo (show network)
  let networks = iterate (learnBatch eta latticeBatch) network
  networks
    & map (`networkBatchError` latticeBatch)
    & zipWith
      ( \epoch errorValue ->
          echo
            ( show epoch
                ++ ": error = "
                ++ show errorValue
            )
      )
      [1 .. epochCount]
    & foldl1 (>>)
  echo ""
  echo "Last state of network:"
  echo $ show $ networks !! epochCount
  hClose fileHandle

handleMonteCarlo :: FilePath -> Batch -> Batch -> NeuralNetwork -> IO ()
handleMonteCarlo filePath latticeBatch randomBatch network = do
  fileHandle <- openFile filePath AppendMode
  let echo = hPutStrLn fileHandle
  echo "First state of network:"
  echo (show network)
  let networks = iterate (learnBatch eta randomBatch) network
  networks
    & map (\nw -> (nw `networkBatchError` latticeBatch, nw `networkBatchError` randomBatch))
    & zipWith
      ( \epoch (testError, learningError) ->
          echo
            ( show epoch
                ++ ": learning error = "
                ++ show learningError
                ++ "; test error = "
                ++ show testError
            )
      )
      [1 .. epochCount]
    & foldl1 (>>)
  echo ""
  echo "Last state of network:"
  echo $ show $ networks !! epochCount
  hClose fileHandle

handleStructure :: Int -> Batch -> Batch -> NetworkStructure -> IO ()
handleStructure times latticeBatch randomBatch networkStructure =
  [ do
      let fileName = (networkStructure & structureName) ++ show x
      rn <- getRandomNetwork 0.1 networkStructure
      handleLattice ('L' : fileName) latticeBatch rn
      handleMonteCarlo ('M' : fileName) latticeBatch randomBatch rn
      putStrLn ((networkStructure & structureName) ++ show x)
    | x <- [1 .. times]
  ]
    & foldl1 (>>)

targetFunction :: Double -> Double -> Double
targetFunction x y = (sin (x - y) - 2 * sin x) * 2 + y

eta :: Double
eta = 0.00002

epochCount :: Int
epochCount = 1000

main :: IO ()
main = do
  randomBatch <- monteCarloSamples
  let latticeBatch = latticeSamples
  allStructures 3 4 [Periodic, No]
    & map (handleStructure 3 latticeBatch randomBatch)
    & foldl1 (>>)
