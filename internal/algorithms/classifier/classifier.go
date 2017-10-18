package classifier

import (
	"github.com/pantsing/hector/internal/algorithms/classifier/ann"
	"github.com/pantsing/hector/internal/algorithms/classifier/common"
	"github.com/pantsing/hector/internal/algorithms/classifier/dt"
	"github.com/pantsing/hector/internal/algorithms/classifier/fm"
	"github.com/pantsing/hector/internal/algorithms/classifier/lr"
	"github.com/pantsing/hector/internal/algorithms/classifier/sa"
	"github.com/pantsing/hector/internal/algorithms/classifier/svm"
	"github.com/pantsing/hector/internal/algorithms/eval"
	"github.com/pantsing/hector/internal/algorithms/internal"
	"github.com/pantsing/hector/internal/core"
	"github.com/pantsing/log"
	"github.com/urfave/cli"
	"math/rand"
	"os"
	"strconv"
	"time"
)

func Commands() []cli.Command {
	cmds := make([]cli.Command, 0, 1<<5)
	for _, alog := range classifierIndex {
		cmd := alog.Command()
		if _, ok := internal.AlogCmdsChecker[cmd.Name]; ok {
			continue
		}
		internal.AlogCmdsChecker[cmd.Name] = struct{}{}
		cmd.Flags = append(cmd.Flags, common.ClassifierCammandFlags...)
		cmd.Action = AlgorithmRun
		cmds = append(cmds, cmd)
	}
	for _, alog := range multiClassClassifierIndex {
		cmd := alog.Command()
		if _, ok := internal.AlogCmdsChecker[cmd.Name]; ok {
			continue
		}
		internal.AlogCmdsChecker[cmd.Name] = struct{}{}
		cmd.Flags = append(cmd.Flags, common.ClassifierCammandFlags...)
		cmd.Action = MultiClassRun
		cmds = append(cmds, cmd)
	}
	return cmds
}

type Classifier interface {
	internal.Algorithm
	//Train model on a given dataset
	Train(dataset *core.DataSet)
	//Predict the probability of a sample to be positive sample
	Predict(sample *core.Sample) float64
	SaveModel(path string)
	LoadModel(path string)
}

var classifierIndex map[string]Classifier = map[string]Classifier{
	"logRegr":         new(lr.LogisticRegression),
	"ftrl":            new(lr.FTRLLogisticRegression),
	"ep":              new(lr.EPLogisticRegression),
	"rdt":             new(dt.RandomDecisionTree),
	"cart":            new(dt.CART),
	"cart-regression": new(dt.RegressionTree),
	"rf":              new(dt.RandomForest),
	"fm":              new(fm.FactorizeMachine),
	"sa":              new(sa.SAOptAUC),
	"gbdt":            new(dt.GBDT),
	"svm":             new(svm.SVM),
	"linear_svm":      new(svm.LinearSVM),
	"l1vm":            new(svm.L1VM),
	"knn":             new(svm.KNN),
	"ann":             new(ann.NeuralNetwork),
	"lrowlqn":         new(lr.LROWLQN),
}

func GetClassifier(method string) Classifier {
	rand.Seed(time.Now().UTC().UnixNano())
	return classifierIndex[method]
}

type MultiClassClassifier interface {
	internal.Algorithm
	//Train model on a given dataset
	Train(dataset *core.DataSet)
	//Predict the probability of a sample to be positive sample
	PredictMultiClass(sample *core.Sample) *core.ArrayVector
	SaveModel(path string)
	LoadModel(path string)
}

var multiClassClassifierIndex map[string]MultiClassClassifier = map[string]MultiClassClassifier{
	"rf":   new(dt.RandomForest),
	"cart": new(dt.CART),
	"rdt":  new(dt.RandomDecisionTree),
	"knn":  new(svm.KNN),
	"ann":  new(ann.NeuralNetwork),
}

func GetMutliClassClassifier(method string) MultiClassClassifier {
	rand.Seed(time.Now().UTC().UnixNano())
	return multiClassClassifierIndex[method]
}

func AlgorithmRun(ctx *cli.Context) (err error) {
	algoName := ctx.Command.Name
	trainSetPath := ctx.String("trainSet")
	testSetPath := ctx.String("testSet")
	predictResultPath := ctx.String("predict")
	modelPath := ctx.String("model")
	global := ctx.Int64("global")
	cv := ctx.Int("cv")

	log.Info("Running ", algoName)
	classifier := GetClassifier(algoName)
	classifier.Init(ctx)

	var trainSet *core.DataSet
	if trainSetPath != "" {
		trainSet = core.NewDataSet()
		err = trainSet.Load(trainSetPath, global)
		if err != nil {
			log.Error(err)
			return
		}
	}

	var testSet *core.DataSet
	if testSetPath != "" {
		testSet = core.NewDataSet()
		err = testSet.Load(testSetPath, global)
		if err != nil {
			log.Error(err)
			return
		}
	}

	var existModel bool
	if modelPath != "" && trainSet == nil && testSet != nil {
		_, err = os.Stat(modelPath)
		existModel = !os.IsNotExist(err)
		if !existModel {
			log.Error(err)
			return err
		}
		classifier.LoadModel(modelPath)
	}

	var predictions []*eval.LabelPrediction
	var auc float64
	if cv <= 1 {
		auc, predictions = AlgorithmRunOnDataSet(classifier, trainSet, testSet)
		if predictions != nil {
			er := eval.ErrorRate(predictions)
			log.Infof("AUC: %.20g\n", auc)
			log.Infof("ER: %.20g\n", er)
		}
	} else {
		average_auc := 0.0
		average_er := 0.0
		for part := 0; part < cv; part++ {
			cvTrainSet, cvTestSet := trainSet.CVSplit(cv, part)
			auc, predictions = AlgorithmRunOnDataSet(classifier, cvTrainSet, cvTestSet)
			log.Infof("AUC: %.20g", auc)
			average_auc += auc
			er := eval.ErrorRate(predictions)
			log.Infof("ER: %.20g", er)
			average_er += er
			classifier.Clear()
		}
		log.Infof("AVG. AUC: %.20g", average_auc/float64(cv))
		log.Infof("AVG. ER: %.20g", average_er/float64(cv))
	}

	if trainSet != nil && modelPath != "" {
		classifier.SaveModel(modelPath)
	}

	if predictResultPath != "" {
		predictResultFile, err := os.Create(predictResultPath)
		if err != nil {
			return err
		}
		defer predictResultFile.Close()

		for i := range predictions {
			predictResultFile.WriteString(strconv.FormatFloat(predictions[i].Prediction, 'g', 5, 64) + "\n")
		}
	}
	return
}

func AlgorithmRunOnDataSet(classifier Classifier, trainSet, testSet *core.DataSet) (float64, []*eval.LabelPrediction) {
	if trainSet != nil {
		classifier.Train(trainSet)
	}

	if testSet == nil {
		return 0.5, nil
	}

	predictions := []*eval.LabelPrediction{}
	for _, sample := range testSet.Samples {
		prediction := classifier.Predict(sample)
		predictions = append(predictions, &(eval.LabelPrediction{Label: sample.Label, Prediction: prediction}))
	}

	auc := eval.AUC(predictions)
	return auc, predictions
}

func MultiClassRun(ctx *cli.Context) (err error) {
	alogName := ctx.Command.Name
	classifier := GetMutliClassClassifier(alogName)
	trainSetPath := ctx.String("trainSet")
	testSetPath := ctx.String("testSet")
	predictResultPath := ctx.String("predict")
	modelPath := ctx.String("model")
	global := ctx.Int64("global")
	cv := ctx.Int("cv")

	classifier.Init(ctx)

	var trainSet *core.DataSet
	if trainSetPath != "" {
		trainSet = core.NewDataSet()
		err = trainSet.Load(trainSetPath, global)
		if err != nil {
			return
		}
	}

	var testSet *core.DataSet
	if testSetPath != "" {
		testSet = core.NewDataSet()
		err = testSet.Load(testSetPath, global)
		if err != nil {
			return
		}
	}

	_, err = os.Stat(modelPath)
	existModel := !os.IsNotExist(err)
	if existModel {
		return err
	}
	classifier.LoadModel(modelPath)

	var predictLabels []int
	var accuracy float64
	if cv <= 1 {
		accuracy, predictLabels = MultiClassRunOnDataSet(classifier, trainSet, testSet)
		log.Infof("Accuracy: %.20g", accuracy)
	} else {
		average_accuracy := 0.0
		for part := 0; part < cv; part++ {
			cvTrainSet, cvTestSet := trainSet.CVSplit(cv, part)
			accuracy, predictLabels = MultiClassRunOnDataSet(classifier, cvTrainSet, cvTestSet)
			log.Infof("Accuracy: %.20g", accuracy)
			average_accuracy += accuracy
			classifier.Clear()
		}
		log.Infof("AVG. Accuracy: %.20g", average_accuracy/float64(cv))
	}

	if trainSet != nil && !existModel {
		classifier.SaveModel(modelPath)
	}

	if predictResultPath != "" {
		predictResultFile, err := os.Create(predictResultPath)
		if err != nil {
			return err
		}
		defer predictResultFile.Close()

		for i := range predictLabels {
			predictResultFile.WriteString(strconv.Itoa(predictLabels[i]) + "\n")
		}
	}
	return
}

func MultiClassRunOnDataSet(classifier MultiClassClassifier, trainSet, testSet *core.DataSet) (accuracy float64, predictLabels []int) {
	if trainSet != nil {
		classifier.Train(trainSet)
	}

	if testSet == nil {
		return 0, nil
	}

	total := 0.0
	predictLabels = make([]int, 0, len(testSet.Samples))
	for _, sample := range testSet.Samples {
		prediction := classifier.PredictMultiClass(sample)
		label, _ := prediction.KeyWithMaxValue()
		predictLabels = append(predictLabels, label)
		if label == sample.Label {
			accuracy += 1.0
		}
		total += 1.0
	}
	accuracy = accuracy / total
	return
}
