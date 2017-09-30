package main

import (
	"runtime"
	"github.com/pantsing/log"
	"github.com/pantsing/hector"
	"github.com/pantsing/hector/eval"
)

func main() {
	train, test, pred, method, params := hector.PrepareParams()

	action, _ := params["action"]

	classifier := hector.GetClassifier(method)
	runtime.GOMAXPROCS(runtime.NumCPU())
	if action == "" {
		auc, predictions, _ := hector.AlgorithmRun(classifier, train, test, pred, params)
		log.Printf("AUC: %.20g\n", auc)
		er := eval.ErrorRate(predictions)
		log.Printf("ER: %.20g\n", er)
	} else if action == "train" {
		hector.AlgorithmTrain(classifier, train, params)
	} else if action == "test" {
		auc, predictions, _ := hector.AlgorithmTest(classifier, test, pred, params)
		log.Printf("AUC: %.20g\n", auc)
		er := eval.ErrorRate(predictions)
		log.Printf("ER: %.20g\n", er)
	}
}
