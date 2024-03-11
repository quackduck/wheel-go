package main

import (
	"encoding/gob"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
)

type layer struct {
	// Weights[i][j] is the weight from neuron i of this layer to neuron j of the previous layer
	// therefore len(Weights) is the number of neurons in this layer and len(Weights[0]) is the number of inputs
	Weights     [][]float64 // Weights for each neuron
	Biases      []float64   // bias for each neuron
	activations []float64   // activations for each neuron
	zs          []float64   // weighted sum for each neuron + bias
	//input       []float64
	//nextLayer *layer
	//prevLayer *layer
}

func newLayer(weights [][]float64, biases []float64) *layer {
	return &layer{
		Weights:     weights,
		Biases:      biases,
		activations: make([]float64, len(weights)),
		zs:          make([]float64, len(weights)),
	}
}

func newRandomLayer(inputSize, thisSize int) *layer {
	return newLayer(randomWeights(inputSize, thisSize), randomBiases(thisSize))
}

func randomWeights(inputSize, thisSize int) [][]float64 {
	weights := make([][]float64, thisSize)
	variance := 2 / float64(inputSize+thisSize)
	//variance := 1 / float64(thisSize)
	for i := range weights {
		weights[i] = make([]float64, inputSize)
		for j := range weights[i] {
			//Weights[i][j] = rand.Float64()*2 - 1
			weights[i][j] = rand.NormFloat64() * math.Sqrt(variance)
		}
	}
	return weights
}

func randomBiases(size int) []float64 {
	biases := make([]float64, size)
	for i := range biases {
		//Biases[i] = (rand.Float64()*2 - 1) * 0
		biases[i] = 0
	}
	return biases
}

//func activate(x float64) float64 {
//	return 1 / (1 + math.Exp(-x))
//}
//
//func activatePrime(x float64) float64 {
//	return activate(x) * (1 - activate(x))
//}

// try tanh()
func activate(x float64) float64 {
	return math.Tanh(x)
}

func activatePrime(x float64) float64 {
	return 1 - math.Tanh(x)*math.Tanh(x)
}

func costFunc(err float64) float64 {
	return err * err
}

func costPrime(err float64) float64 {
	return 2 * err
}

type network struct {
	Layers []*layer
	cost   float64
}

func (n *network) forward(input []float64) []float64 {
	n.Layers[0].activations = input // set the input layer's activations
	for i := range n.Layers {
		if i == 0 {
			continue
		}
		//fmt.Println("forwarding layer", i, "with input", n[i-1].activations, "Weights", n[i].Weights, "Biases", n[i].Biases)
		n.Layers[i].forward(input)
		input = n.Layers[i].activations
	}
	return n.Layers[len(n.Layers)-1].activations
}

func (l *layer) forward(input []float64) {
	for i, weights := range l.Weights { // for each neuron in this layer
		var sum float64
		for j, weight := range weights { // for each input to that neuron
			sum += weight * input[j]
		}
		l.zs[i] = sum + l.Biases[i]
		l.activations[i] = activate(l.zs[i])
	}
}

func (n *network) backward(wanted []float64) {
	n.cost = 0.0
	for i, activation := range n.Layers[len(n.Layers)-1].activations {
		n.cost += costFunc(activation - wanted[i])
	}

	rate := 0.001

	delta := make([]float64, len(wanted)) // delta[j] is delC / delZ[j]
	for i, activation := range n.Layers[len(n.Layers)-1].activations {
		delta[i] = 2 * costPrime(activation-wanted[i]) * activatePrime(n.Layers[len(n.Layers)-1].zs[i]) // delC / delZ
	}
	updateLayerWithDelta(n.Layers[len(n.Layers)-1], n.Layers[len(n.Layers)-2], delta, rate)

	for i := len(n.Layers) - 2; i >= 1; i-- { // for each layer except the input and output
		newdelta := make([]float64, len(n.Layers[i].activations)) // one for each neuron
		// go over all the neurons in the next layer and add up the impact of their delta on this layer's delta
		for j := range n.Layers[i].activations { // for each neuron in this layer
			var sum float64
			for k := range n.Layers[i+1].activations { // for each neuron in the next layer
				sum += n.Layers[i+1].Weights[k][j] * delta[k]
			}
			newdelta[j] = sum * activatePrime(n.Layers[i].zs[j])
		}
		updateLayerWithDelta(n.Layers[i], n.Layers[i-1], newdelta, rate)
		delta = newdelta
	}
}

type pair struct {
	input  []float64
	output []float64
}

func (n *network) backwardBatched(pairs []pair) {
	allDeltas := make([][]float64, len(n.Layers))
	for i := range allDeltas { // for each layer
		allDeltas[i] = make([]float64, len(n.Layers[i].Weights)) // one for each neuron
	}

	// add cost for all pairs
	n.cost = 0.0

	for _, p := range pairs {
		n.forward(p.input)

		for i, activation := range n.Layers[len(n.Layers)-1].activations {
			n.cost += costFunc(activation - p.output[i])
		}

		delta := make([]float64, len(n.Layers[len(n.Layers)-1].activations)) // delta[j] is delC / delZ[j]
		for i, activation := range n.Layers[len(n.Layers)-1].activations {
			delta[i] = 2 * costPrime(activation-p.output[i]) * activatePrime(n.Layers[len(n.Layers)-1].zs[i]) // delC / delZ
		}
		//fmt.Println("delta", allDeltas[len(n.Layers)-1])
		sum1d(allDeltas[len(n.Layers)-1], delta) // add the delta to the last layer's deltas
		//fmt.Println("delta 2", allDeltas[len(n.Layers)-1])

		for i := len(n.Layers) - 2; i >= 1; i-- { // for each layer except the input and output
			newdelta := make([]float64, len(n.Layers[i].activations)) // one for each neuron
			//fmt.Println("newdelta", newdelta, "delta", delta)
			// go over all the neurons in the next layer and add up the impact of their delta on this layer's delta
			for j := range n.Layers[i].activations { // for each neuron in this layer
				var sum float64
				for k := range n.Layers[i+1].activations { // for each neuron in the next layer
					sum += n.Layers[i+1].Weights[k][j] * delta[k]
				}
				newdelta[j] = sum * activatePrime(n.Layers[i].zs[j])
			}
			sum1d(allDeltas[i], newdelta)
			delta = newdelta
		}
	}

	n.cost /= float64(len(pairs))

	rate := 0.001 / float64(len(pairs))

	for i := len(n.Layers) - 1; i >= 1; i-- { // for each layer except the input
		//multScalar1d(1/float64(len(pairs)), allDeltas[i])
		updateLayerWithDelta(n.Layers[i], n.Layers[i-1], allDeltas[i], rate)
		//fmt.Println("deltas", allDeltas[i])
	}
}

// sum1d adds two 1d slices together and stores the result in the first slice
func sum1d(one []float64, two []float64) {
	for i := range one {
		one[i] += two[i]
	}
}

func multScalar1d(scalar float64, arr []float64) {
	for i := range arr {
		arr[i] *= scalar
	}
}

func updateLayerWithDelta(l *layer, lp *layer, delta []float64, learnStepRate float64) {
	for i, weights := range l.Weights {
		for j := range weights {
			l.Weights[i][j] -= learnStepRate * errorToDelCDelWeight(delta[i], lp.activations[j])
		}
	}
	for i := range l.Biases {
		l.Biases[i] -= learnStepRate * errorToDelCDelBias(delta[i])
	}
}

// delta should be the error for the neuron that this weight is for
// prevActive should be the activation of the neuron that this weight is for in the previous layer
func errorToDelCDelWeight(delta, prevActive float64) float64 {
	return delta * prevActive
}

// delta should be the error for the neuron that this bias is for
func errorToDelCDelBias(delta float64) float64 {
	return delta
}

func (n *network) String() string {
	result := ""
	for i, layer := range n.Layers {
		result += "Layer " + fmt.Sprint(i) + ":\n"
		result += "Weights: " + fmt.Sprint(layer.Weights) + "\n"
		result += "Biases: " + fmt.Sprint(layer.Biases) + "\n\n"
	}
	return result
}

func saveToFile(n *network, filename string) {
	// use gob
	file, err := os.Create(filename)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	err = encoder.Encode(n)
	if err != nil {
		panic(err)
	}
}

func loadFromFile(filename string) *network {
	// use gob
	file, err := os.Open(filename)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	var n network
	err = decoder.Decode(&n)
	if err != nil {
		panic(err)
	}
	for i := range n.Layers {
		n.Layers[i].activations = make([]float64, len(n.Layers[i].Weights))
		n.Layers[i].zs = make([]float64, len(n.Layers[i].Weights))
	}
	return &n
}
