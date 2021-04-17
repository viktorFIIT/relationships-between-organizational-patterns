library(tidyverse)
library(tidytext)

patterns <- read.delim("data/patterns.txt", sep="@")
patterns <- patterns[, -1]

words <- tibble(description = patterns$Description) %>% unnest_tokens(word, description)
words %>%
  inner_join(get_sentiments("bing")) %>% 
  count(sentiment) %>% 
  spread(sentiment, n, fill = 0) %>% 
  mutate(sentiment = positive - negative)

getSentiment <- function(description){
  tokens <- data_frame(text = description) %>% unnest_tokens(word, text)
  sentiment <- tokens %>%
    inner_join(get_sentiments("bing")) %>%
    count(sentiment) %>% 
    spread(sentiment, n, fill = 0)
  return (sentiment)
}

size <- length(patterns$Description)
value<-vector(mode="character")
sentiments <- data.frame("negative", "positive")
for(i in 1:size){
  s <- getSentiment(patterns[[2]][[i]])
  if (!is.null(s$negative) && !is.null(s$positive)) { 
    sentiments[i, 1] = s$negative
    sentiments[i, 2] = s$positive
  }
}

ngramator <- function(description) { 
  return (lapply(ngrams(words(description), numberOfGrams), paste, collapse = " "))
}

numberOfGrams = 2

library(tm)

(patternssCorpus <- VCorpus(VectorSource(unlist(lapply(patterns$Description, as.character)))))
bigram_matrix <- DocumentTermMatrix((patternssCorpus), control = list(tokenize = ngramator, stopwords = FALSE, stemming = TRUE))
(bigram_freq <- sort(colSums(as.matrix(bigram_matrix)), decreasing=TRUE))

top_bigrams <- function(bigrams, most){
  top_bigram_list <- c()
  for(bigram in bigrams){
    unigrams <- strsplit(bigram, " ")
    if(!(bigrams[[1]][1] %in% stopwords("en") | bigrams[[1]][2]  %in% stopwords("en"))){
      top_bigram_list <- c(top_bigram_list, bigram) 
    }
    if (length(top_bigram_list) == most){
      break
    }
  }
  return (top_bigram_list)
}

most_used_bigrams <- top_bigrams(names(bigram_freq), 30)
dtm_bigram_df <- data.frame(as.matrix(bigram_matrix[, intersect(colnames(bigram_matrix), most_used_bigrams)]))
dtm_bigram_df$patternName <- patterns$Name
cat_freq <- dtm_bigram_df %>% group_by(patternName) %>% summarise_each(funs(sum))

library(reshape2)

ggplot(melt(cat_freq),aes(x=variable, y=value, fill = patternName)) + 
  geom_col(position = "dodge") + coord_flip() + xlab("unigrams") + ylab("bigrams_frequency") + 
  theme() + ggtitle("Most used words across pattern descriptions") + ggsave("graphs/most-used-words-across-patterns.pdf") 

bigramTokenizer <- function(x){
  NGramTokenizer(x, Weka_control(min=2, max=2))
}

cleanCorpus <- function(text){
  text.tmp <- tm_map(text, removePunctuation)
  text.tmp <- tm_map(text.tmp, stripWhitespace)
  text.tmp <- tm_map(text.tmp, content_transformer(tolower))
  text.tmp <- tm_map(text.tmp, removeNumbers)
  return(text.tmp)
}

frequentBigrams <- function(text){
  cleanCorpus <- cleanCorpus(VCorpus(VectorSource(text)))
  dtm <- TermDocumentMatrix(cleanCorpus, control=list(tokenize=bigramTokenizer))
  word_freqs <- sort(rowSums(as.matrix(dtm)), decreasing=T)
  return (dm <- data.frame(word=names(word_freqs), freq=word_freqs))
}

text <- intersect(patterns$Associations, patterns$Associations)

library(igraph)
library(RWeka)
library(ggraph)

bigrams_network <- frequentBigrams(text) %>% 
  separate(word, c("word1", "word2"), sep=" ") %>%
  filter(freq > 2) %>% 
  graph_from_data_frame()

set.seed(123)
arrows <- grid::arrow(type="closed", length=unit(0.15, "inches"))

ggraph(bigrams_network, layout="fr") +
  geom_edge_link(aes(edge_alpha=freq), show.legend = F, arrow=arrows, end_cap=circle(0.07,'inches')) +
  geom_edge_density(aes(fill = freq)) +
  geom_node_point(color="#99F1EB", size=5) +
  geom_node_text(aes(label=name), repel=T) +
  theme_void() +
  labs(title="Connections between association sections of org. patterns") +
  ggsave("graphs/connections-between-words.pdf")
