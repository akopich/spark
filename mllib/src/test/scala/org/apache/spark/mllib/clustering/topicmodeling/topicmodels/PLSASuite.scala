package org.apache.spark.mllib.clustering.topicmodeling.topicmodels

import java.util.Random

import org.apache.spark.mllib.clustering.topicmodeling.topicmodels.regulaizers.{SymmetricDirichletDocumentOverTopicDistributionRegularizer, SymmetricDirichletTopicRegularizer}

/**
 * Created by valerij on 6/27/14.
 */
class PLSASuite extends AbstractTopicModelSuite {
    test ("feasibility") {
        val numberOfTopics = 2
        val numberOfIterations = 10

        val plsa = new PLSA(sc,
            numberOfTopics,
            numberOfIterations,
            new Random(),
            new SymmetricDirichletDocumentOverTopicDistributionRegularizer(0.2f),
            new SymmetricDirichletTopicRegularizer(0.2f))

        testPLSA(plsa)
    }

}
