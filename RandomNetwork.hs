module RandomNetwork
  ( getRandomNetwork,
    getURandomNetwork,
    getZeroNetwork,
  )
where

import ActivationFunction ( ActivationFunction )
import qualified Data.ByteString.Lazy as LazyBS
import Data.Function ((&))
import GHC.Float (rationalToDouble)
import Matrix (Matrix (..), limit)
import NeuralNetwork
  ( Layer (..),
    NeuralNetwork (..),
  )
import Random (getRandomDoubles, getURandomDoubles)
import Semiring (Semiring (zero))

fillList :: [a] -> [a] -> ([a], [a])
fillList ds xs = splitAt (length xs) ds

fillMatrix :: [x] -> Matrix x -> (Matrix x, [x])
fillMatrix ds mat =
  rows mat
    & foldl
      ( \(matrix, doubles) nextRow ->
          let (newRow, lastDoubles) = nextRow & fillList doubles
           in (Matrix $ rows matrix ++ [newRow], lastDoubles)
      )
      (Matrix [], ds)

fillLayer :: [Double] -> Layer -> (Layer, [Double])
fillLayer doubles layer = (Layer (fst filledMatrix) (layer & activators), snd filledMatrix)
  where
    filledMatrix = fillMatrix doubles (layer & weights)

fillNetwork :: [Double] -> NeuralNetwork -> NeuralNetwork
fillNetwork ds network =
  layers network
    & foldl
      ( \(net, doubles) nextLayer ->
          let (newLayer, lastDoubles) = nextLayer & fillLayer doubles
           in (NeuralNetwork $ layers net ++ [newLayer], lastDoubles)
      )
      (NeuralNetwork [], ds)
    & fst

getZeroLayer :: Int -> [ActivationFunction] -> Layer
getZeroLayer inputs nodeInfo =
  Layer
    (zero & limit (length nodeInfo) (1 + inputs))
    nodeInfo

getZeroNetwork :: Int -> [[ActivationFunction]] -> NeuralNetwork
getZeroNetwork inputs nodeInfo =
  NeuralNetwork $
    zipWith
      getZeroLayer
      (inputs : map length nodeInfo)
      nodeInfo

getRandomNetwork :: Double -> Int -> [[ActivationFunction]] -> IO NeuralNetwork
getRandomNetwork weightLimit inputs nodeInfo = do
  randoms <- getRandomDoubles weightLimit
  return (getZeroNetwork inputs nodeInfo & fillNetwork randoms)

getURandomNetwork :: Double -> Int -> [[ActivationFunction]] -> IO NeuralNetwork
getURandomNetwork weightLimit inputs nodeInfo = do
  randoms <- getURandomDoubles weightLimit
  return (getZeroNetwork inputs nodeInfo & fillNetwork randoms)