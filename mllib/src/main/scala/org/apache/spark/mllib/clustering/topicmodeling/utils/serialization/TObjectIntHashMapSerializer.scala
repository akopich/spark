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

package org.apache.spark.mllib.clustering.topicmodeling.utils.serialization

import com.esotericsoftware.kryo.io.{Input, Output}
import com.esotericsoftware.kryo.{Kryo, Serializer}
import gnu.trove.map.hash.TObjectIntHashMap

/**
 * this is a serializer for TObjectIntHashMapSerializer, should be passed to register(...) kryo
 * method
 */
class TObjectIntHashMapSerializer extends Serializer[TObjectIntHashMap[Object]] {


  def write(kryo: Kryo, out: Output, map: TObjectIntHashMap[Object]) {
    val values = map.values
    val keys = map.keys()
    kryo.writeObject(out, values)
    kryo.writeObject(out, keys)
  }

  def read(kryo: Kryo, in: Input, clazz: Class[TObjectIntHashMap[Object]]):
  TObjectIntHashMap[Object] = {
    val values = kryo.readObject(in, classOf[Array[Int]])
    val keys = kryo.readObject(in, classOf[Array[Object]])

    val map = new TObjectIntHashMap[Object]()

    keys.zip(values).map { case (key, value) => map.put(key, value)
    }
    map
  }

}
