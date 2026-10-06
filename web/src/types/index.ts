export interface HealthStatus {
  status: string;
  app_name: string;
  env: string;
  database_connected: boolean;
  models_ready: Record<string, boolean>;
  dataset_loaded: boolean;
  article_count: number;
  user_count: number;
}

export interface UserSummary {
  user_id: string;
  history_count: number;
  top_categories: string[];
}

export interface EntityPreference {
  name: string;
  wikidata_id?: string;
  count: number;
  category?: string;
}

export interface UserProfile {
  user_id: string;
  history_size: number;
  category_distribution: Record<string, number>;
  subcategory_distribution: Record<string, number>;
  top_entities: EntityPreference[];
  top_categories: string[];
  recent_articles: Array<{
    article_id: string;
    title: string;
    category: string;
    subcategory: string;
  }>;
  semantic_summary: string;
}

export interface FeedItem {
  recommendation_id: string;
  article_id: string;
  title: string;
  abstract: string;
  category: string;
  subcategory: string;
  url?: string;
  rank: number;
  score: number;
  sources: string[];
  primary_reason: string;
}

export interface FeedResponse {
  user_id: string;
  model_used: string;
  total_returned: number;
  items: FeedItem[];
}

export interface RecommendationExplanation {
  recommendation_id: string;
  article_id: string;
  user_id: string;
  rank: number;
  final_score: number;
  sources: string[];
  component_scores: {
    content: number;
    collaborative: number;
    popularity: number;
  };
  feature_breakdown: Record<string, number>;
  primary_reason: string;
  summary_text: string;
}

export interface ModelStatus {
  name: string;
  model_type: string;
  status: string;
  artifact_available: boolean;
  description: string;
}

export interface EvaluationResult {
  model_name: string;
  auc: number;
  mrr: number;
  ndcg_at_5: number;
  ndcg_at_10: number;
  recall_at_k: number;
  coverage: number;
  num_impressions: number;
  evaluated_at: string;
  dataset_source?: string;
  evaluation_mode?: string;
  impression_coverage?: number;
  catalog_coverage?: number;
  hit_rate_at_k?: number;
}
