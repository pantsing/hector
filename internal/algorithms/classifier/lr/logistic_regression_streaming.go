package lr

import (
	"bufio"
	"log"
	"math"
	"os"
	"strconv"
	"strings"

	"github.com/pantsing/hector/internal/core"
	"github.com/pantsing/hector/internal/utils"
	"github.com/urfave/cli"
)

func (algo *LogisticRegressionStream) Command() cli.Command {
	return cli.Command{
		Name:     "streamLogRegr",
		Usage:    "Streaming Logistic Regression",
		Category: "LR",
		Flags: []cli.Flag{
			cli.Float64Flag{
				Name: "learning-rate,lrate",
			},
			cli.Float64Flag{
				Name: "regularization,r",
			},
			cli.IntFlag{
				Name: "steps",
			},
		},
	}
}

type LogisticRegressionStream struct {
	Model  map[int64]float64
	Params LogisticRegressionParams
}

func (algo *LogisticRegressionStream) SaveModel(path string) {
	sb := utils.StringBuilder{}
	for f, g := range algo.Model {
		sb.Int64(f)
		sb.Write("\t")
		sb.Float(g)
		sb.Write("\n")
	}
	sb.WriteToFile(path)
}

func (algo *LogisticRegressionStream) LoadModel(path string) {
	file, _ := os.Open(path)
	defer file.Close()

	scaner := bufio.NewScanner(file)
	for scaner.Scan() {
		line := scaner.Text()
		tks := strings.Split(line, "\t")
		fid, _ := strconv.ParseInt(tks[0], 10, 64)
		fw, _ := strconv.ParseFloat(tks[1], 64)
		algo.Model[fid] = fw
	}
}

func (algo *LogisticRegressionStream) Init(ctx *cli.Context) {
	algo.Model = make(map[int64]float64)
	algo.Params.LearningRate = ctx.Float64("learning-rate")
	algo.Params.Regularization = ctx.Float64("regularization")
	algo.Params.Steps = ctx.Int("steps")
}

func (algo *LogisticRegressionStream) Train(dataset *core.StreamingDataSet) {
	algo.Model = make(map[int64]float64)
	totalErr := 0.0
	n := 0
	for sample := range dataset.Samples {
		prediction := algo.Predict(sample)
		err := sample.LabelDoubleValue() - prediction
		totalErr += math.Abs(err)
		n += 1
		if n%100000 == 0 {
			log.Println("proc ", n, totalErr/100000.0, sample.LabelDoubleValue(), prediction)
			totalErr = 0.0
		}
		for _, feature := range sample.Features {
			model_feature_value, ok := algo.Model[feature.Id]
			if !ok {
				model_feature_value = 0.0
			}
			model_feature_value += algo.Params.LearningRate * (err*feature.Value - algo.Params.Regularization*model_feature_value)
			algo.Model[feature.Id] = model_feature_value
		}
	}
}

func (algo *LogisticRegressionStream) Predict(sample *core.Sample) float64 {
	ret := 0.0
	for _, feature := range sample.Features {
		model_feature_value, ok := algo.Model[feature.Id]
		if ok {
			ret += model_feature_value * feature.Value
		}
	}
	return utils.Sigmoid(ret)
}
