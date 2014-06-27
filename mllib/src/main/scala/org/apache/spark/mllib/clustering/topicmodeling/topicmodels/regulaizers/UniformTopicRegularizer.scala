package org.apache.spark.mllib.clustering.topicmodeling.topicmodels.regulaizers

/**
 * Created by valerij on 6/26/14.
 */

/**
 * usage of this prior is equivalent to use of no prior
 */
class UniformTopicRegularizer extends TopicsRegularizer {
    override def apply(topics: Array[Array[Float]]): Float = 0

    override def regularize(topics: Array[Array[Float]], oldTopics: Array[Array[Float]]): Unit = {}
}
