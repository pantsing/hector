package core

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/pantsing/hector/internal/utils"
	"io"
)

type CombinedFeature []string

type FeatureSplit []float64

func FindCategory(split []float64, value float64) int {
	return sort.Search(len(split), func(i int) bool { return split[i] >= value })
}

/* RawDataSet */
type RawDataSet struct {
	Samples     []*RawSample
	FeatureKeys map[string]bool
}

func NewRawDataSet() *RawDataSet {
	ret := RawDataSet{}
	ret.Samples = []*RawSample{}
	ret.FeatureKeys = make(map[string]bool)
	return &ret
}

func (d *RawDataSet) AddSample(sample *RawSample) {
	if sample == nil {
		return
	}
	d.Samples = append(d.Samples, sample)
}

func (d *RawDataSet) ToDataSet(splits map[string][]float64, combinations []CombinedFeature) *DataSet {
	out_data := NewDataSet()
	fm := make(map[string]int64)
	for _, sample := range d.Samples {
		out_sample := NewSample()
		out_sample.Label = sample.Label
		if splits != nil {
			for fkey_str, fvalue_str := range sample.Features {
				fkey := ""
				fvalue := 0.0
				if GetFeatureType(fkey_str) == FeatureTypeEnum.CONTINUOUS_FEATURE {
					split, ok := splits[fkey_str]
					if ok {
						cat := FindCategory(split, utils.ParseFloat64(fvalue_str))
						fkey = fkey_str + "_" + strconv.FormatInt(int64(cat), 10)
						fvalue = 1.0
					} else {
						fvalue = utils.ParseFloat64(fvalue_str)
					}
					fm[fkey] = utils.Hash(fkey)
					out_sample.AddFeature(Feature{Id: utils.Hash(fkey), Value: fvalue})
				}
			}
		}
		for _, combination := range combinations {
			fkey := ""
			for _, ckey := range combination {
				fkey += ckey
				fkey += ":"
				fkey += sample.GetFeatureValue(ckey)
				fkey += "_"
			}
			fm[fkey] = utils.Hash(fkey)
			out_sample.AddFeature(Feature{Id: utils.Hash(fkey), Value: 1.0})
		}
		out_data.AddSample(out_sample)
	}
	f, _ := os.Create("features.tsv")
	defer f.Close()
	w := bufio.NewWriter(f)
	for k, v := range fm {
		w.WriteString(k + "\t" + strconv.FormatInt(v, 10) + "\n")
	}

	return out_data
}

func (d *RawDataSet) Load(path string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	ch := make(chan string, 1000)
	go func() {
		reader := bufio.NewReader(file)
		for {
			line, err := reader.ReadString('\n')
			if err != nil {
				break
			}
			ch <- line
		}
		close(ch)
	}()

	n := 0
	for line := range ch {
		n += 1
		if n%10000 == 0 {
			fmt.Println(n, len(ch))
		}
		line = strings.Replace(line, " ", "\t", -1)
		tks := strings.Split(line, "\t")
		sample := NewRawSample()
		for i, tk := range tks {
			if i == 0 {
				label, err := strconv.ParseInt(tk, 10, 16)
				if err != nil {
					break
				}
				if label > 0 {
					sample.Label = 1.0
				} else {
					sample.Label = 0.0
				}
			} else {
				kv := strings.Split(tk, ":")
				sample.Features[kv[0]] = kv[1]
				d.FeatureKeys[kv[0]] = true
			}
		}
		d.AddSample(sample)
	}
	return nil
}

/*Streaming*/
type StreamingDataSet struct {
	Samples chan *Sample
}

func NewStreamingDataSet() *StreamingDataSet {
	return &StreamingDataSet{
		Samples: make(chan *Sample, 10000),
	}
}

func (d *StreamingDataSet) AddSample(sample *Sample) {
	if sample == nil {
		return
	}
	d.Samples <- sample
}

func (d *StreamingDataSet) Load(path string, globalBiasFeatureID int64) error {
	file, err := os.Open(path)
	defer file.Close()
	if err != nil {
		log.Fatalln("load file fail: ", err)
	}
	reader := bufio.NewReader(file)
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			break
		}
		sample := d.parser(line, globalBiasFeatureID)
		d.AddSample(sample)
	}
	close(d.Samples)
	return nil
}

func (d *StreamingDataSet) LoadFromStdIn(globalBiasFeatureID int64) error {
	reader := bufio.NewReader(os.Stdin)
	for {
		line, err := reader.ReadString('\n')
		if err == io.EOF {
			break
		}
		sample := d.parser(line, globalBiasFeatureID)
		d.AddSample(sample)
	}
	close(d.Samples)
	return nil
}

func (d *StreamingDataSet) parser(line string, globalBiasFeatureID int64) (sample *Sample) {
	tks := strings.Split(strings.TrimSpace(line), "\t")
	sample = &Sample{Features: make([]Feature, 0, 20), Label: 0}
	if globalBiasFeatureID >= 0 {
		sample.Features = append(sample.Features, Feature{globalBiasFeatureID, 1.0})
	}
	for i, tk := range tks {
		if i == 0 {
			label, _ := strconv.Atoi(tk)
			sample.Label = label
		} else {
			if strings.TrimSpace(tk) == "" {
				continue
			}
			kv := strings.Split(tk, ":")
			feature_id, err := strconv.ParseInt(kv[0], 10, 64)
			if err != nil {
				log.Println("wrong feature: ", tk)
				return nil
			}
			feature_value := 1.0
			if len(kv) > 1 {
				feature_value, err = strconv.ParseFloat(kv[1], 64)
				if err != nil {
					log.Println("wrong value: ", tk)
					return nil
				}
			}
			feature := Feature{feature_id, feature_value}
			sample.Features = append(sample.Features, feature)
		}
	}
	return sample
}

/* DataSet */
type DataSet struct {
	Samples          []*Sample
	FeatureNameIdMap map[int64]string
	max_label        int
}

func NewDataSet() *DataSet {
	ret := DataSet{}
	ret.Samples = []*Sample{}
	ret.FeatureNameIdMap = make(map[int64]string)
	return &ret
}

func (d *DataSet) AddSample(sample *Sample) {
	d.Samples = append(d.Samples, sample)
	if d.max_label < sample.Label {
		d.max_label = sample.Label
	}
}

func (d *DataSet) Load(path string, globalBiasFeatureID int64) error {
	fm := make(map[string]int64)

	ch := make(chan string, 1000)
	go func() {
		file, err := os.Open(path)
		defer file.Close()
		defer close(ch)
		if err != nil {
			log.Println("load file fail: ", err)
			return
		}

		scanner := bufio.NewScanner(file)

		for scanner.Scan() {
			line := strings.Replace(scanner.Text(), " ", "\t", -1)
			ch <- line
		}
	}()

	for line := range ch {
		tks := strings.Split(strings.TrimSpace(line), "\t")
		sample := Sample{Features: make([]Feature, 0, 20), Label: 0}
		if globalBiasFeatureID >= 0 {
			sample.Features = append(sample.Features, Feature{globalBiasFeatureID, 1.0})
		}
		for i, tk := range tks {
			if i == 0 {
				label, _ := strconv.Atoi(tk)
				sample.Label = label
				if d.max_label < label {
					d.max_label = label
				}
			} else {
				kv := strings.Split(tk, ":")
				feature_id, err := strconv.ParseInt(kv[0], 10, 64)
				if err != nil {
					feature_id = utils.Hash(kv[0])
					fm[kv[0]] = feature_id
				}
				d.FeatureNameIdMap[feature_id] = kv[0]
				feature_value := 1.0
				if len(kv) > 1 {
					feature_value, err = strconv.ParseFloat(kv[1], 64)
					if err != nil {
						break
					}
				}
				feature := Feature{feature_id, feature_value}
				sample.Features = append(sample.Features, feature)
			}
		}
		//if globalBiasFeatureID >= 0 {
		//	sample.Features = append(sample.Features, Feature{globalBiasFeatureID, 1.0})
		//}
		d.AddSample(&sample)
	}
	f, _ := os.Create("features.tsv")
	defer f.Close()
	w := bufio.NewWriter(f)
	for k, v := range fm {
		w.WriteString(k + "\t" + strconv.FormatInt(v, 10) + "\n")
	}

	log.Println("dataset size : ", len(d.Samples))
	return nil
}

func RemoveLowFreqFeatures(dataset *DataSet, threshold float64) {
	freq := NewVector()

	for _, sample := range dataset.Samples {
		for _, feature := range sample.Features {
			freq.AddValue(feature.Id, 1.0)
		}
	}

	for _, sample := range dataset.Samples {
		features := []Feature{}
		for _, feature := range sample.Features {
			if freq.GetValue(feature.Id) > threshold {
				features = append(features, feature)
			}
		}
		sample.Features = features
	}
}

func (d *DataSet) Split(f func(int) bool) *DataSet {
	out_data := NewDataSet()
	for i, sample := range d.Samples {
		if f(i) {
			out_data.AddSample(sample)
		}
	}
	return out_data
}

func (d *DataSet) CVSplit(cvTotal, cvPart int) (trainSet *DataSet, testSet *DataSet) {
	trainSet = NewDataSet()
	testSet = NewDataSet()

	for i, sample := range d.Samples {
		if i%cvTotal == cvPart {
			testSet.AddSample(sample)
		} else {
			testSet.AddSample(sample)
		}
	}
	return
}

/* Real valued DataSet */
type RealDataSet struct {
	Samples []*RealSample
}

func NewRealDataSet() *RealDataSet {
	ret := RealDataSet{}
	ret.Samples = []*RealSample{}
	return &ret
}

func (d *RealDataSet) AddSample(sample *RealSample) {
	d.Samples = append(d.Samples, sample)
}

func (d *RealDataSet) Load(path string, globalBiasFeatureID int64) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		line := strings.Replace(scanner.Text(), " ", "\t", -1)
		tks := strings.Split(line, "\t")
		sample := RealSample{Features: []Feature{}, Value: 0.0}
		for i, tk := range tks {
			if i == 0 {
				value := utils.ParseFloat64(tk)
				sample.Value = value
			} else {
				kv := strings.Split(tk, ":")
				feature_id, err := strconv.ParseInt(kv[0], 10, 64)
				if err != nil {
					break
				}
				feature_value := 1.0
				if len(kv) > 1 {
					feature_value, err = strconv.ParseFloat(kv[1], 64)
					if err != nil {
						break
					}
				}
				feature := Feature{feature_id, feature_value}
				sample.Features = append(sample.Features, feature)
			}
		}
		if globalBiasFeatureID >= 0 {
			sample.Features = append(sample.Features, Feature{globalBiasFeatureID, 1.0})
		}
		d.AddSample(&sample)
	}
	if scanner.Err() != nil {
		return scanner.Err()
	}
	return nil
}

func (d *RealDataSet) CVSplit(cvTotal, cvPart int) (trainSet *RealDataSet, testSet *RealDataSet) {
	trainSet = NewRealDataSet()
	testSet = NewRealDataSet()

	for i, sample := range d.Samples {
		if i%cvTotal == cvPart {
			testSet.AddSample(sample)
		} else {
			testSet.AddSample(sample)
		}
	}
	return
}
