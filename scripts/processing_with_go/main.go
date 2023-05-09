package main

import (
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
)

type Config struct {
    Header     string `json:"header"`
    Entities []Entity  `json:"named_entity"`
}

type Entity struct {
	Title interface{} `json:"title"`
	SubTitle interface{} `json:"subtitle"`
	Contents []Content `json:"content"`
}

type Content struct {
	Sentence string `json:"sentence"`
	Labels interface{} `json:"labels"`
}
 

func loading_json_file(filepath string) []string {
	jsonFile, err := os.Open(filepath)
	if err != nil {
		fmt.Println(err)
	}
	fmt.Printf("Successfully Opened %s\n", filepath)
	// defer the closing of our jsonFile so that we can parse it later on
	defer jsonFile.Close()

    byteValue, _ := ioutil.ReadAll(jsonFile)

	var config Config
    json.Unmarshal([]byte(byteValue), &config)

	var entities = config.Entities

	var results []string 

	var re1 = regexp.MustCompile(`(\w+)+@\(이메일\)\sⓒ\s\(이메일\)`)
	var re2 = regexp.MustCompile(`(\w+)+@`)
	var re3 = regexp.MustCompile(`[a-zA-Z0-9+-_.]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+`)
	var re4 = regexp.MustCompile(`\([^)]*\)`)

	for _, entity := range entities {
		target_entity := entity.Contents
		var context []string
		// length := len(target_entity)
		for _, content := range target_entity {
			var result = content.Sentence

			var ignore_sign = []string{"관련기사", "참조링크"}
			for _, sign := range ignore_sign{
				index := strings.Index(result, sign)
				if index != -1{
					result = ""
					break
				}
			}
			
			start_index := strings.Index(result, "(이름) 기자")
			if start_index != -1{
				result = result[start_index+10:]
			} 

			context = append(context, result)


		}
		results = append(results, strings.Join(context, " "))
	}

	return results
}
 
func main() {
	wait := new(sync.WaitGroup)

	files, err := filepath.Glob("../030.웹데이터 기반 한국어 말뭉치 데이터/01.데이터/1.Training/라벨링데이터/TL1/*/*.json")
	if err != nil {
		// Handle error
		fmt.Println("oh?")
	}

	wait.Add(len(files))
	
	var results []string
	for _, file := range files {
		go func(file string) {
			result := loading_json_file(file)
			results = append(results, result...)
			defer wait.Done() 
		}(file)
	}
	wait.Wait()

	f5, _ := os.Create("./tmp/corpus.txt")
	_, _ = io.WriteString(f5, strings.Join(results, "\n\n=====\n\n"))
}