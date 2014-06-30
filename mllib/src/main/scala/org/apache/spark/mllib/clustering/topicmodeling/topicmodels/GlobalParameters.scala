package org.apache.spark.mllib.clustering.topicmodeling.topicmodels

/**
 * Created by valerij on 6/30/14.
 *
 */
/**
 *
 * @param phi -- distribution of topics over words
 * @param alphabetSize
 */
class GlobalParameters(val phi : Array[Array[Float]], val alphabetSize : Int)
