package main

import (
	"github.com/petar/GoMNIST"
	"math/rand"
)

func makeTrainingDataMNIST() (inputs, targets [][]float64) {
	train, _, err := GoMNIST.Load("./data")
	if err != nil {
		panic(err)
	}

	size := 10000

	inputs = make([][]float64, size)
	targets = make([][]float64, size)

	offset := rand.Intn(60000 - size) // only use the first 6000 images for now

	for i := 0; i < size; i++ {
		img, label := train.Get(offset + i)
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

func makeTestingDataMNIST() (inputs, targets [][]float64) {
	_, test, err := GoMNIST.Load("./data")
	if err != nil {
		panic(err)
	}

	inputs = make([][]float64, 1000)
	targets = make([][]float64, 1000)
	for i := 0; i < 1000; i++ {
		img, label := test.Get(i)
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
