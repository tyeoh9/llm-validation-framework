class toxicityAgent:
    def deterministicLayer(self, statement, custom_bad_words = []):
      # You can load your own custom "Illegal" list from TCS
      profanity.load_censor_words(custom_bad_words)
      if profanity.contains_profanity(statement):
          return "Illegal/Toxic Content Detected"
      return 'Okay Statement'

    def probabilisticLayer(self, statement):
      model = Detoxify('original')

      results = model.predict(statement)
      return results

    def semanticLayer(self,statement, illegal_categories, threshold = 0.5):
      model = SentenceTransformer('all-MiniLM-L6-v2')
      deny_embeddings = model.encode(illegal_categories, convert_to_tensor=True)

      def semantic_validation(user_prompt, threshold=0.6):

          user_embedding = model.encode(user_prompt, convert_to_tensor=True)


          cosine_scores = util.cos_sim(user_embedding, deny_embeddings)

          # Find the highest similarity score
          max_score = torch.max(cosine_scores).item()

          # Determine the "Grade"
          if max_score > threshold:
              return "FAIL", max_score
          return "PASS", max_score

      # Test it out

      status, score = semantic_validation(statement,threshold)
      return status, score
    def evaluation(self,statement, custom_bad_words = [], illegal_categories = [], threshold = 0.5):
      if illegal_categories == []:
        return 'Must have at least 1 category'
      layer_one = self.deterministicLayer(statement, custom_bad_words)
      layer_two = self.probabilisticLayer(statement)
      layer_three = self.semanticLayer(statement, illegal_categories, threshold)
      return {'layer_one':layer_one,'layer_two':layer_two,'layer_three':layer_three}