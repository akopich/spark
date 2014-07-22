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

package org.apache.spark.mllib.clustering.topicmodeling.documents

import breeze.linalg.SparseVector
import gnu.trove.map.hash.TObjectIntHashMap
import org.apache.spark.SparkContext.rddToPairRDDFunctions
import org.apache.spark.rdd.RDD


/**
 * This object numerates tokens. E.g. it replaces a word with its order number. It also
 * calculates the number of unique words
 */
object Enumerator {

  /**
   *
   * @param rawDocuments RDD of tokenized documents (every document is a sequence of tokens
   *                     (Strings) )
   * @param rareTokenThreshold tokens that are encountered in the collection less than
   *                           rareTokenThreshold times are omitted
   * @return RDD of documents with tokens replaced with their order numbers
   */
  def numerate(rawDocuments: RDD[Seq[String]], rareTokenThreshold: Int) = {
    val alphabet = rawDocuments.context.broadcast(getAlphabet(rawDocuments, rareTokenThreshold))

    rawDocuments.map(document => mkDocument(document, alphabet.value))
  }


  private def mkDocument(rawDocument: Seq[String], alphabet: TObjectIntHashMap[String]) = {
    val wordsMap = rawDocument.map(alphabet.get).foldLeft(Map[Int, Int]().withDefaultValue(0))(
      (map, word) => map + (word -> (1 + map(word))))

    val words = wordsMap.keys.toArray.sorted

    val tokens = new SparseVector[Int](words, words.map(word => wordsMap(word)), alphabet.size())
    new Document(tokens, alphabet.size())
  }

  private def getAlphabet(rawDocuments: RDD[Seq[String]], rareTokenThreshold: Int) = {
    val alphabet = new TObjectIntHashMap[String]()

    rawDocuments.flatMap(x => x).map(x => (x, 1)).reduceByKey(_ + _).filter(_._2 >
      rareTokenThreshold).collect.map(_._1).zipWithIndex.foreach { case (key,
    value) => alphabet.put(key, value)
    }
    alphabet
  }
}
