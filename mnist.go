package main

import (
	"fmt"
	"github.com/petar/GoMNIST"
	"math/rand/v2"
)

var trainData, testData *GoMNIST.Set
var indices []int

func init() {
	var err error
	trainData, testData, err = GoMNIST.Load("./data")
	if err != nil {
		panic(err)
	}
	indices = rand.Perm(60000) // shuffle
}

var currIndex int

func makeTrainingDataMNIST() (inputs, targets [][]float64) {
	size := 6000 // batch size

	inputs = make([][]float64, size)
	targets = make([][]float64, size)

	for i := 0; i < size; i++ {
		img, label := trainData.Get(indices[(i+currIndex)%60000]) // random order
		inputs[i] = make([]float64, 28*28)
		targets[i] = make([]float64, 10)
		for j := 0; j < 28*28; j++ {
			inputs[i][j] = float64(img[j]) / 255 // normalize
		}

		for j := 0; j < 10; j++ {
			targets[i][j] = -1
		}
		targets[i][label] = 1
	}
	currIndex += size
	if currIndex == 60000 {
		fmt.Println("Epoch finished")
		indices = rand.Perm(60000) // shuffle
	}
	currIndex %= 60000 // this is optional because the index wraps around anyway
	return
}

func makeTestingDataMNIST() (inputs, targets [][]float64) {
	size := 1000

	inputs = make([][]float64, size)
	targets = make([][]float64, size)
	for i := 0; i < size; i++ {
		img, label := testData.Get(i)
		//fmt.Println(len(img), label)
		inputs[i] = make([]float64, 28*28)
		targets[i] = make([]float64, 10)
		for j := 0; j < 28*28; j++ {
			inputs[i][j] = float64(img[j]) / 255 // normalize
		}

		for j := 0; j < 10; j++ {
			targets[i][j] = -1
		}
		targets[i][label] = 1
	}
	return
}
