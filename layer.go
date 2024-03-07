package main

import (
	"math"
	"math/rand"
)

type layer struct {
	// weights[i][j] is the weight from neuron i of this layer to neuron j of the previous layer
	// therefore len(weights) is the number of neurons in this layer and len(weights[0]) is the number of inputs
	weights     [][]float64 // weights for each neuron
	biases      []float64   // bias for each neuron
	activations []float64   // activations for each neuron
	zs          []float64   // weighted sum for each neuron + bias
	//input       []float64

	//nextLayer *layer
	//prevLayer *layer
}

func newLayer(weights [][]float64, biases []float64) *layer {
	return &layer{
		weights:     weights,
		biases:      biases,
		activations: make([]float64, len(weights)),
		zs:          make([]float64, len(weights)),
	}
}

func newRandomLayer(inputSize, thisSize int) *layer {
	return newLayer(randomWeights(inputSize, thisSize), randomBiases(thisSize))
}

func randomWeights(inputSize, thisSize int) [][]float64 {
	weights := make([][]float64, thisSize)
	for i := range weights {
		weights[i] = make([]float64, inputSize)
		for j := range weights[i] {
			weights[i][j] = rand.Float64()*2 - 1
		}
	}
	return weights
}

func randomBiases(size int) []float64 {
	biases := make([]float64, size)
	for i := range biases {
		biases[i] = rand.Float64()*2 - 1
	}
	return biases
}

func activate(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func activatePrime(x float64) float64 {
	return activate(x) * (1 - activate(x))
}

func costFunc(err float64) float64 {
	return err * err
}

func costPrime(err float64) float64 {
	return 2 * err
}

type network []*layer

func (n network) forward(input []float64) []float64 {
	n[0].activations = input // set the input layer's activations
	for i := range n {
		if i == 0 {
			continue
		}
		//fmt.Println("forwarding layer", i, "with input", n[i-1].activations, "weights", n[i].weights, "biases", n[i].biases)
		n[i].forward(input)
		input = n[i].activations
	}
	return n[len(n)-1].activations
}

func (l *layer) forward(input []float64) {
	for i, weights := range l.weights { // for each neuron in this layer
		var sum float64
		for j, weight := range weights { // for each input to that neuron
			sum += weight * input[j]
		}
		l.zs[i] = sum + l.biases[i]
		l.activations[i] = activate(l.zs[i])
	}
}

func (n network) backward(wanted []float64) {

	rate := 0.1

	delta := make([]float64, len(wanted)) // delta[j] is delC / delZ[j]
	for i, activation := range n[len(n)-1].activations {
		delta[i] = 2 * costPrime(activation-wanted[i]) * activatePrime(n[len(n)-1].zs[i]) // delC / delZ
		updateLayerWithDelta(n[len(n)-1], n[len(n)-2], delta, rate)
	}

	for i := len(n) - 2; i >= 1; i-- { // for each layer except the input and output
		newdelta := make([]float64, len(n[i].activations)) // one for each neuron
		// go over all the neurons in the next layer and add up the impact of their delta on this layer's delta
		for j := range n[i].activations { // for each neuron in this layer
			var sum float64
			for k := range n[i+1].activations { // for each neuron in the next layer
				sum += n[i+1].weights[k][j] * delta[k]
			}
			newdelta[j] = sum * activatePrime(n[i].zs[j])
		}
		updateLayerWithDelta(n[i], n[i-1], newdelta, rate)
		delta = newdelta
	}
}

func updateLayerWithDelta(l *layer, lp *layer, delta []float64, learnRate float64) {
	for i, weights := range l.weights {
		for j := range weights {
			l.weights[i][j] -= learnRate * errorToDelCDelWeight(delta[i], lp.activations[j])
		}
	}
	for i := range l.biases {
		l.biases[i] -= learnRate * errorToDelCDelBias(delta[i])
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
