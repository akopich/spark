/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.clustering.topicmodeling.topicmodels

import java.util.Random

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.clustering.topicmodeling.documents.Document
import org.apache.spark.mllib.clustering.topicmodeling.topicmodels.regulaizers.{DocumentOverTopicDistributionRegularizer, TopicsRegularizer, UniformDocumentOverTopicRegularizer, UniformTopicRegularizer}
import org.apache.spark.rdd.RDD
import org.apache.spark.{Logging, SparkContext}


/**
 * Created by valerij on 6/25/14.
 */
/**
 *
 * distributed topic modeling via PLSA (Hofmann (1999), Vorontsov, Potapenko (2014) )
 * @param sc  spark context
 * @param numberOfTopics number of topics
 * @param numberOfIterations number of iterations
 * @param random java.util.Random need for initialisation
 * @param documentOverTopicDistributionRegularizer
 * @param topicRegularizer
 * @param computePpx boolean. If true, model computes perplexity and prints it puts in the log at
 *                   INFO level. it takes some time and memory
 */
class PLSA(@transient protected val sc: SparkContext,
           protected val numberOfTopics: Int,
           private val numberOfIterations: Int,
           protected val random: Random,
           private val documentOverTopicDistributionRegularizer:
              DocumentOverTopicDistributionRegularizer = new UniformDocumentOverTopicRegularizer,
           @transient protected val topicRegularizer: TopicsRegularizer =
                  new UniformTopicRegularizer,
           private val computePpx: Boolean = true)
  extends AbstractPLSA[DocumentParameters, GlobalParameters, GlobalCounters]
  with Logging
  with Serializable {

  override def infer(documents: RDD[Document]): (RDD[DocumentParameters], GlobalParameters) = {
    val alphabetSize = getAlphabetSize(documents)

    val collectionLength = getCollectionLength(documents)

    val topicBC = getInitialTopics(alphabetSize)

    val parameters = documents.map(doc => DocumentParameters(doc,
                                                    numberOfTopics,
                                                    documentOverTopicDistributionRegularizer))

    val (result, topics) = newIteration(parameters, topicBC, alphabetSize, collectionLength, 0)

    (result, new GlobalParameters(topics.value,alphabetSize))
  }


  private def newIteration(parameters: RDD[DocumentParameters],
       topicsBC: Broadcast[Array[Array[Float]]],
       alphabetSize: Int,
       collectionLength: Int,
       numberOfIteration: Int): (RDD[DocumentParameters], Broadcast[Array[Array[Float]]]) = {
    if (computePpx) {
      logInfo("Interation number " + numberOfIteration)
      logInfo("Perplexity=" + perplexity(topicsBC, parameters, collectionLength))
    }
    if (numberOfIteration == numberOfIterations) {
      (parameters, topicsBC)
    } else {
      val newParameters = parameters.map(param => param.getNewTheta(topicsBC)).cache()
      val globalCounters = getGlobalCounters(parameters, topicsBC, alphabetSize)
      val newTopics = getTopics(newParameters, alphabetSize, topicsBC.value, globalCounters)

      parameters.unpersist()

      newIteration(newParameters,
        sc.broadcast(newTopics),
        alphabetSize,
        collectionLength,
        numberOfIteration + 1)
    }
  }

  private def getGlobalCounters(parameters: RDD[DocumentParameters],
                                  topics: Broadcast[Array[Array[Float]]], alphabetSize: Int) = {
    parameters.aggregate[GlobalCounters](GlobalCounters(numberOfTopics,
      alphabetSize))((thatOne, otherOne) => thatOne.add(otherOne, topics, alphabetSize),
        (thatOne, otherOne) => thatOne + otherOne)
  }

  private def perplexity(topicsBC: Broadcast[Array[Array[Float]]],
                         parameters: RDD[DocumentParameters], collectionLength: Int) = {
    generalizedPerplexity(topicsBC,
      parameters,
      collectionLength,
      par => (word,num) => num * math.log(probabilityOfWordGivenTopic(word, par, topicsBC)).toFloat)
  }
}
