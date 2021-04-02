module NeuralNetworkTest where

import ActivationFunction ( ActivationFunction(No, Gauss) )
import BackPropagation ( learnBatch )
import Data.Function ((&))
import NeuralNetwork
    ( applyNetwork,
      networkBatchError,
      Batch,
      Input,
      NetworkStructure(NetworkStructure),
      NeuralNetwork,
      Output )
import RandomNetwork ( getRandomNetwork )

batch :: Batch
batch =
  [ ([1, 0], [0]),
    ([0, 1], [0]),
    ([1, 1], [1]),
    ([0, 0], [1])
  ]

networkStructure :: NetworkStructure
networkStructure =
  NetworkStructure
    2
    [ [Gauss, Gauss],
      [No]
    ]

learnBatchNTimes :: Int -> Double -> [(Input, Output)] -> NeuralNetwork -> NeuralNetwork
learnBatchNTimes n eta batch network =
  iterate (learnBatch eta batch) network
    !! n

printNetworkInfo :: NeuralNetwork -> [(Input, Output)] -> IO ()
printNetworkInfo network batch = do
  putStr $ "networkError : " ++ show (networkBatchError network batch) ++ "\n"
  putStr "listOfErrors: \n"
  putStr
    ( ( map (applyNetwork network . fst) batch
          & zipWith
            ( \sample result ->
                show (sample & fst) ++ "__" ++ show result ++ "__" ++ "(" ++ show (sample & snd) ++ ")\n"
            )
            batch
      )
        & concat
    )

main :: IO ()
main = do
  network <-
    getRandomNetwork
      0.5
      networkStructure
  print network
  printNetworkInfo network batch
  putStr "\n\n\n"
  let newNetwork = learnBatchNTimes 4000 0.3 batch network
  print newNetwork
  printNetworkInfo newNetwork batch
  return ()