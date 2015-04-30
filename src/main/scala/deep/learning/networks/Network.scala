package deep.learning.networks

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.sigmoid

import scala.util.Random

/**
 * Created by inakov on 4/30/15.
 */
class Network(networkDefinition: DenseVector[Int]) {

  val numberOfLayers: Int = networkDefinition.length
  val definition: DenseVector[Int] = networkDefinition

  val biases: DenseVector[DenseVector[Double]] = {
    for(y <- definition(1 to -1))
      yield DenseVector.rand[Double](y)
  }

  val weights: DenseVector[DenseMatrix[Double]] = {
    val values = for((x, y) <- definition(0 to -2).valuesIterator.zip(definition(1 to -1).valuesIterator).toArray)
      yield DenseMatrix.rand[Double](x, y)
    DenseVector[DenseMatrix[Double]](values)
  }

  println(biases)
  println(weights)

  def feedforward(a: DenseVector[Double]) ={
    var result = a;
    for((b, w) <- biases.valuesIterator.zip(weights.valuesIterator))
      result = sigmoid((w * result) + b)
    result
  }

  def SGD(trainingData: List, epochs: Int, miniBatchSize: Int, eta: Double): Unit ={
    val trainingDataLength: Int = trainingData.length
    for(j <- 0 to epochs){
      Random.shuffle(trainingData)
      val miniBatches: List[List] =
        for(k <- (0 to trainingDataLength by miniBatchSize).toList) yield trainingData.slice(k, k + miniBatchSize)

    }
  }

//  def SGD(self, training_data, epochs, mini_batch_size, eta,
//          test_data=None):
//  """Train the neural network using mini-batch stochastic
//        gradient descent.  The "training_data" is a list of tuples
//        "(x, y)" representing the training inputs and the desired
//        outputs.  The other non-optional parameters are
//        self-explanatory.  If "test_data" is provided then the
//        network will be evaluated against the test data after each
//        epoch, and partial progress printed out.  This is useful for
//        tracking progress, but slows things down substantially."""
//  if test_data: n_test = len(test_data)
//  n = len(training_data)
//  for j in xrange(epochs):
//    random.shuffle(training_data)
//  mini_batches = [
//  training_data[k:k+mini_batch_size]
//  for k in xrange(0, n, mini_batch_size)]
//  for mini_batch in mini_batches:
//    self.update_mini_batch(mini_batch, eta)
//  if test_data:
//    print "Epoch {0}: {1} / {2}".format(
//    j, self.evaluate(test_data), n_test)
//  else:
//  print "Epoch {0} complete".format(j)
}
