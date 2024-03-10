package main

import (
	"fmt"
	"math"
)

func main() {

	//n := &network{
	//	Layers: []*layer{
	//		newRandomLayer(0, 784), // input layer
	//		//newRandomLayer(784, 32),
	//		newRandomLayer(784, 16),
	//		newRandomLayer(16, 16),
	//		newRandomLayer(16, 16),
	//		//newRandomLayer(16, 16),
	//		//newRandomLayer(16, 16),
	//		//newRandomLayer(16, 16),
	//		//newRandomLayer(16, 16),
	//		//newRandomLayer(32, 10),
	//		newRandomLayer(16, 10), // output layer
	//		//newRandomLayer(4, 5),
	//		//newRandomLayer(10, 1),
	//	},
	//}

	filename := "model5.gob"

	n := loadFromFile(filename)

	//fmt.Println(n)

	//fmt.Println(n[1].Weights)

	//input := []float64{1}
	//fmt.Println(n.forward(input))

	//fmt.Println(n)

	// train
	lastCost := -1.0

	//fmt.Println("Training data size", len(inputs))
	for i := 0; i < 1000; {
		i++ // start at 1
		inputs, targets := makeTrainingDataMNIST()
		//for d := 0.0; d < 1; d += 0.01 {
		//	//input := []float64{d}
		//	//target := []float64{math.Sin(d * 2 * math.Pi)}
		//	//target := []float64{d * d}
		//	//if input[0] > 0.5 {
		//	//target[0] = input[0]
		//	//}
		//	n.forward(input)
		//	n.backward(target)
		//}
		for i := range inputs {
			n.forward(inputs[i])
			n.backward(targets[i])
		}

		if i%10 == 0 {
			fmt.Println("iteration", i, "cost", n.cost)
			if lastCost > 0 {
				fmt.Println("cost ratio", n.cost/lastCost)
				fmt.Println("improvement", 1/n.cost-1/lastCost)
				fmt.Println("Score", 1/n.cost)
			}
			lastCost = n.cost

			test(n)

			//for d := 0.0; d < 1; d += 0.01 {
			//	input := []float64{d}
			//	fmt.Println(input[0], ",", n.forward(input)[0])
			//	//fmt.Println(n.forward(input)[0])
			//}
			//if i > 0 {
			//	return
			//}
		}

		if i%100 == 0 {
			saveToFile(n, filename)
			fmt.Println("Saved model to", filename, "\n")
		}

		//if n.cost < 1e-10 {
		//	fmt.Println("Found a good enough solution")
		//	break
		//}

		//input := []float64{0.5}
		//target := []float64{0.5}
		//n.forward(input)
		//n.backward(target)
		//
		//input = []float64{0.6}
		//target = []float64{0.1}
		//n.forward(input)
		//n.backward(target)
	}

	//for d := -1.0; d < 1; d += 0.005 {
	//	input := []float64{d}
	//	fmt.Println(input[0], ",", n.forward(input)[0])
	//	//fmt.Println(n.forward(input)[0])
	//}

	fmt.Println("Score", 1/n.cost)

	test(n)

	//fmt.Println(n.Layers[1].Weights)

	//fmt.Println(n)
}

func test(n *network) {
	inputs, targets := makeTestingDataMNIST()
	//fmt.Println("Testing data size", len(inputs))
	correct := 0
	for i := range inputs {
		output := n.forward(inputs[i])
		//fmt.Println("Output", output, "Target", targets[i])
		if getIndexWithMaxValue(output) == getIndexWithMaxValue(targets[i]) {
			correct++
		} else {
			//fmt.Println("Incorrect")
		}
	}
	fmt.Println("Correct", correct, "out of", len(inputs))
}

func oneHot(n int, i int) []float64 {
	out := make([]float64, n)
	out[i] = 1.0
	return out
}

func getIndexWithMaxValue(arr []float64) int {
	max := 0.0
	index := 0
	for i, v := range arr {
		if v > max {
			max = v
			index = i
		}
	}
	return index
}

func makeTrainingDataSin() (inputs, targets [][]float64) {
	inputs = make([][]float64, 0, 100)
	targets = make([][]float64, 0, 100)
	for d := -1.0; d < 1; d += 0.01 {
		inputs = append(inputs, []float64{d})
		targets = append(targets, []float64{math.Sin(d * 2 * math.Pi)})
	}
	return
}

func makeTrainingDataX2() (inputs, targets [][]float64) {
	inputs = make([][]float64, 0, 100)
	targets = make([][]float64, 0, 100)
	for d := -1.0; d < 1; d += 0.01 {
		inputs = append(inputs, []float64{d})
		targets = append(targets, []float64{d * d})
	}
	return
}

func makeTrainingDataNegX() (inputs, targets [][]float64) {
	inputs = make([][]float64, 0, 100)
	targets = make([][]float64, 0, 100)
	for d := -1.0; d < 1; d += 0.01 {
		inputs = append(inputs, []float64{d})
		targets = append(targets, []float64{-d})
	}
	return
}
