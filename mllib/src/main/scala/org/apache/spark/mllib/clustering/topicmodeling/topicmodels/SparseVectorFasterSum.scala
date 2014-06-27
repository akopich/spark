package org.apache.spark.mllib.clustering.topicmodeling.topicmodels

import breeze.linalg.SparseVector

/**
 * Created by valerij on 6/25/14.
 */
trait SparseVectorFasterSum {
    protected def sum[T](vector: SparseVector[Short]) = {
        var sum = 0
        for (element <- vector) sum += element
        sum
    }
}
