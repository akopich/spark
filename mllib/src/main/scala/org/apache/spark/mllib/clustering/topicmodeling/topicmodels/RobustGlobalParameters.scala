package org.apache.spark.mllib.clustering.topicmodeling.topicmodels

/**
 * Created by valerij on 6/30/14.
 */
class RobustGlobalParameters(phi : Array[Array[Float]],
                             alphabetSize: Int,
                             background : Array[Float] ) extends GlobalParameters(phi, alphabetSize)
