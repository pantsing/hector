package common

import "github.com/urfave/cli"

var Commands []cli.Command = make([]cli.Command, 0, 1<<3)

var ClassifierCammandFlags []cli.Flag = []cli.Flag{
	//cli.StringFlag{
	//	Name:  "action, a",
	//	Value: "run",
	//	Usage: `"run", "train", "test" or "predict"`,
	//},
	cli.IntFlag{
		Name:  "crossValidation,cv",
		Value: 1,
		Usage: "Cross Validation",
	},
	cli.StringFlag{
		Name: "trainSet, train",
	},
	cli.StringFlag{
		Name: "testSet, test",
	},
	cli.StringFlag{
		Name: "predictResult, predict",
	},
	cli.StringFlag{
		Name:  "modelPath, model",
		Usage: "If trained on data set and model path is set, output model to the path.",
	},
	cli.IntFlag{
		Name:  "globalBiasFeatureID,global",
		Value: 0,
		Usage: "If you read/write a model file， you MUST set the global bias feature ID.",
	},
}
