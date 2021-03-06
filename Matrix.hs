{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
module Matrix where

import Data.Function ((&))
import Data.List (transpose)
import Semiring (Semiring (..))
import Control.DeepSeq (NFData, force)
import GHC.Generics (Generic)

newtype Matrix x = Matrix {rows :: [[x]]} deriving (Eq, NFData, Generic)

instance Functor Matrix where
  fmap f (Matrix m) = Matrix $ map (map f) m

instance Applicative Matrix where
  pure = Matrix . repeat . repeat
  funcs <*> m =
    zipWith
     (zipWith ($))
     (rows funcs)
     (rows m)
    & Matrix

cols :: Matrix x -> [[x]]
cols (Matrix m) = transpose m

transposeMatrix :: Matrix x -> Matrix x
transposeMatrix m = m & cols & Matrix

listProd :: (Semiring x, NFData x) => [x] -> [x] -> x
listProd xs ys =
  zipWith prod xs ys
    & foldl plus zero
    & force

buildMatrix :: (a -> b -> c) -> [a] -> [b] -> Matrix c
buildMatrix f as bs = Matrix $ map (\a -> map (f a) bs) as

instance (Semiring x, NFData x) => Semiring (Matrix x) where
  zero = Matrix {rows = repeat $ repeat zero}
  one =
    [0, 1 ..]
      & map (\n -> replicate n zero ++ (one : repeat zero))
      & Matrix
  plus (Matrix m1) (Matrix m2) =
    Matrix $
      zipWith (zipWith plus) m1 m2
  prod m1 m2 = buildMatrix listProd (rows m1) (cols m2)

instance (Show x) => Show (Matrix x) where
  show m1 = rows m1 & concatMap (\row -> concatMap (\num -> show num ++ " ") row ++ "\n")

limit :: Int -> Int -> Matrix x -> Matrix x
limit maxRows maxCols (Matrix m) = Matrix $ take maxRows $ map (take maxCols) m