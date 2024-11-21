use cetana::{tensor::Tensor, MlError, MlResult};
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

pub struct DataLoader {
    data: Vec<Vec<usize>>,
    batch_size: usize,
    context_length: usize,
    current_pos: usize,
    indices: Vec<usize>,
}

impl DataLoader {
    pub fn new(data_path: &str, batch_size: usize, context_length: usize) -> MlResult<Self> {
        // if data_path is not a file, download the data
        if !Path::new(data_path).exists() {
            Self::download_shakespeare_data(data_path)?;
        }

        // Load text
        let file = File::open(Path::new(data_path)).map_err(|e| {
            MlError::StringError(format!("Failed to open file: {}", e))
        })?;
        let reader = BufReader::new(file);
        let text: String = reader
            .lines()
            .filter_map(Result::ok)
            .collect::<Vec<String>>()
            .join("\n");

        // Create vocabulary (char-level tokenization)
        let chars: Vec<char> = text
            .chars()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        let vocab_size = chars.len();

        // Create lookup tables
        let stoi: std::collections::HashMap<char, usize> =
            chars.iter().enumerate().map(|(i, &c)| (c, i)).collect();

        // Encode the text
        let data: Vec<usize> = text.chars().map(|c| *stoi.get(&c).unwrap()).collect();

        println!("Vocabulary size: {}", vocab_size);
        println!("Data length: {}", data.len());

        // Create training chunks
        let mut chunked_data = Vec::new();
        for chunk in data.windows(context_length + 1) {
            if chunk.len() == context_length + 1 {
                chunked_data.push(chunk.to_vec());
            }
        }

        let mut indices: Vec<usize> = (0..chunked_data.len()).collect();
        indices.shuffle(&mut thread_rng());

        Ok(Self {
            data: chunked_data,
            batch_size,
            context_length,
            current_pos: 0,
            indices,
        })
    }

    pub fn iter(&mut self) -> MlResult<DataLoaderIterator> {
        Ok(DataLoaderIterator {
            loader: self,
            current_batch: 0,
        })
    }

    pub fn num_batches(&self) -> usize {
        self.data.len() / self.batch_size
    }

    pub fn get_batch(&mut self, batch_idx: usize) -> MlResult<(Tensor, Tensor)> {
        println!("Creating batch {}", batch_idx);
        let start_idx = batch_idx * self.batch_size;
        let end_idx = (batch_idx + 1) * self.batch_size;
        println!("Batch indices: {}..{}", start_idx, end_idx);

        let mut input_batch = Vec::new();
        let mut target_batch = Vec::new();

        for idx in start_idx..end_idx {
            if idx >= self.indices.len() {
                println!("Warning: idx {} >= indices len {}", idx, self.indices.len());
                break;
            }
            let data_idx = self.indices[idx];
            if data_idx >= self.data.len() {
                println!(
                    "Warning: data_idx {} >= data len {}",
                    data_idx,
                    self.data.len()
                );
                continue;
            }

            let sequence = &self.data[data_idx];
            input_batch.push(sequence[..self.context_length].to_vec());
            target_batch.push(sequence[1..=self.context_length].to_vec());
        }

        println!("Created batch with {} sequences", input_batch.len());

        Ok((
            Tensor::new(
                input_batch
                    .into_iter()
                    .map(|v| v.into_iter().map(|i| i as f32).collect())
                    .collect(),
            )?,
            Tensor::new(
                target_batch
                    .into_iter()
                    .map(|v| v.into_iter().map(|i| i as f32).collect())
                    .collect(),
            )?,
        ))
    }

    pub fn get_eval_batch(&mut self) -> MlResult<(Tensor, Tensor)> {
        // Get a random batch for evaluation
        let batch_idx = rand::random::<usize>() % self.num_batches();
        self.get_batch(batch_idx)
    }

    pub fn shuffle(&mut self) {
        self.indices.shuffle(&mut thread_rng());
        self.current_pos = 0;
    }

    pub fn download_shakespeare_data(output_path: &str) -> MlResult<()> {
        let url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt";

        // Create a blocking client for the download
        let response = reqwest::blocking::get(url)
            .map_err(|e| MlError::StringError(format!("Failed to download data: {}", e)))?;

        let content = response
            .text()
            .map_err(|e| MlError::StringError(format!("Failed to get content: {}", e)))?;

        // Create the output file
        let mut file = File::create(output_path)
            .map_err(|e| MlError::StringError(format!("Failed to create file: {}", e)))?;

        // Write the content
        // file.write_all(content.as_bytes())
        // .map_err(|e| MlError::StringError(format!("Failed to write data: {}", e)))?;

        Ok(())
    }
}

pub struct DataLoaderIterator<'a> {
    loader: &'a mut DataLoader,
    current_batch: usize,
}

impl<'a> Iterator for DataLoaderIterator<'a> {
    type Item = MlResult<(Tensor, Tensor)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_batch >= self.loader.num_batches() {
            self.loader.shuffle();
            self.current_batch = 0;
            None
        } else {
            let batch = self.loader.get_batch(self.current_batch);
            self.current_batch += 1;
            Some(batch)
        }
    }
}
