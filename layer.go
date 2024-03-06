package main

import (
	"math"
	"math/rand"
)

type layer struct {
	// weights[i][j] is the weight from neuron i of input to neuron j of this layer
	// therefore weights has size
	weights     [][]float64
	biases      []float64
	activations []float64
	gradients   []float64

	nextLayer *layer
	prevLayer *layer
}

func newLayer(weights [][]float64, biases []float64) *layer {
	return &layer{
		weights:     weights,
		biases:      biases,
		activations: make([]float64, len(biases)),
		gradients:   make([]float64, len(biases)),
	}
}

func newRandomLayer(inputSize, thisSize int) *layer {
	return newLayer(randomWeights(inputSize, thisSize), randomBiases(thisSize))
}

func (l *layer) connectNext(nextLayer *layer) {
	l.nextLayer = nextLayer
}

func (l *layer) connectPrev(prevLayer *layer) {
	l.prevLayer = prevLayer
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

func (l *layer) forward(input []float64) {
	l.activations = make([]float64, len(l.weights))
	for i, weights := range l.weights {
		var sum float64
		for j, weight := range weights {
			sum += weight * input[j]
		}
		l.activations[i] = activate(sum + l.biases[i])
	}

	if l.nextLayer != nil {
		l.nextLayer.forward(l.activations)
	}
}

var costList []float64

// assumes forward has been called and activations are set
func (l *layer) backward(wanted []float64) {

	if l.prevLayer == nil {
		// this is the first layer, so we don't need to update anything
		return
	}

	learnRate := 0.1

	//// get current cost
	var cost float64
	for i, activation := range l.activations {
		cost += costFunc(activation - wanted[i])
	}
	//fmt.Println("cost:", cost)
	costList = append(costList, cost)

	// calculate derivative of cost with respect to this layer's weights.
	// we do this by getting derivative of cost with respect to activations (dC/dA)
	// then derivative of activations with respect to weighted sum (dA/dZ)
	// then derivative of weighted sum with respect to weights (dZ/dW)
	// multiply these together to get dC/dW (chain rule)

	// cost = sum(costFunc(sum(activate(weights * input + bias)) - wanted))
	// z = w * input + bias
	// a = activate(z)
	// therefore cost = sum(costFunc(a - wanted))

	// dC/dA -> impact of each activation on the cost
	dCdA := make([]float64, len(l.activations))
	for i, activation := range l.activations {
		dCdA[i] = costPrime(activation - wanted[i])
	}

	// dA/dZ
	dAdZ := make([]float64, len(l.activations))
	for i, activation := range l.activations {
		dAdZ[i] = activatePrime(activation)
	}

	// dZ/dW
	dZdW := make([][]float64, len(l.weights))
	for i, weights := range l.weights { // for each neuron
		dZdW[i] = make([]float64, len(weights))
		for j := range weights { // for each input of that neuron
			dZdW[i][j] = l.prevLayer.activations[j]
		}
	}

	// dC/dW
	dCdW := make([][]float64, len(l.weights))
	for i, weights := range l.weights { // for each neuron
		dCdW[i] = make([]float64, len(weights))
		for j := range weights { // for each input of that neuron
			dCdW[i][j] = dCdA[i] * dAdZ[i] * dZdW[i][j]
		}
	}

	// update weights
	for i, weights := range l.weights {
		for j := range weights {
			l.weights[i][j] -= learnRate * dCdW[i][j] // negative gradient
		}
	}

	// update biases
	for i := range l.biases {
		l.biases[i] -= learnRate * dCdA[i] * dAdZ[i] // negative gradient
	}

	// dc/daprev = sum(dc/dz * dz/daprev)
	// dz/daprev = weights
	// dc/dz = sum(dc/da * da/dz)
	// da/dz = activatePrime

	dCdAprev := make([]float64, len(l.prevLayer.activations))

	for i := range l.prevLayer.activations {
		var sum float64
		for j, weights := range l.weights { // for each neuron
			sum += dCdA[j] * dAdZ[j] * weights[i]
		}
		dCdAprev[i] = sum
	}

	// now we can figure out what the activations of the previous layer should have been
	// and update them

	activationsWanted := make([]float64, len(l.prevLayer.activations))
	for i := range l.prevLayer.activations {
		activationsWanted[i] = l.prevLayer.activations[i] - dCdAprev[i]*learnRate // negative gradient
	}

	//if l.prevLayer != nil {
	//	l.prevLayer.backward(activationsWanted)
	//}
}
