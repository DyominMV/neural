module Random
  ( getRandomDoubles,
    getURandomDoubles,
  )
where

import qualified Data.ByteString.Lazy as LazyBS
import Data.Function ((&))
import GHC.Float (rationalToDouble)

applyPairs :: (a -> a -> b) -> [a] -> [b]
applyPairs f [] = []
applyPairs f [_] = []
applyPairs f (a1 : a2 : as) = f a1 a2 : applyPairs f as

getDoublesFromFile :: String -> Double -> IO [Double]
getDoublesFromFile fileName limit = do
  randomBytes <- LazyBS.readFile fileName
  return
    ( ( LazyBS.unpack randomBytes
          & map toInteger
          & applyPairs (\a b -> a * 256 + b)
          & applyPairs (\a b -> a * 256 * 256 + b)
      )
        & map (((* limit) . (\d -> d * 2 -1)) . (`rationalToDouble` (256 * 256 * 256 * 256 - 1)))
    )

getRandomDoubles :: Double -> IO [Double]
getRandomDoubles = getDoublesFromFile "/dev/random"

getURandomDoubles :: Double -> IO [Double]
getURandomDoubles = getDoublesFromFile "/dev/urandom"
