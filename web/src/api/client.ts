import {
  HealthStatus,
  UserSummary,
  UserProfile,
  FeedResponse,
  RecommendationExplanation,
  ModelStatus,
  EvaluationResult,
} from '../types';

const BASE_URL = '/api';

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let errorDetail = res.statusText;
    try {
      const errJson = await res.json();
      if (errJson.detail) errorDetail = errJson.detail;
      else if (errJson.message) errorDetail = errJson.message;
    } catch {
      // fallback to statusText
    }
    throw new Error(errorDetail);
  }
  return res.json();
}

export async function fetchHealth(): Promise<HealthStatus> {
  const res = await fetch(`${BASE_URL}/health`);
  return handleResponse<HealthStatus>(res);
}

export async function fetchUsers(limit: number = 50): Promise<UserSummary[]> {
  const res = await fetch(`${BASE_URL}/users?limit=${limit}`);
  return handleResponse<UserSummary[]>(res);
}

export async function fetchUserProfile(userId: string): Promise<UserProfile> {
  const res = await fetch(`${BASE_URL}/users/${userId}/profile`);
  return handleResponse<UserProfile>(res);
}

export async function fetchFeed(
  userId: string,
  model: string = 'ranker',
  limit: number = 20,
  diversityLambda?: number,
  exploreEpsilon?: number
): Promise<FeedResponse> {
  let url = `${BASE_URL}/feed/${userId}?model=${model}&limit=${limit}`;
  if (diversityLambda !== undefined) url += `&diversity_lambda=${diversityLambda}`;
  if (exploreEpsilon !== undefined) url += `&explore_epsilon=${exploreEpsilon}`;
  const res = await fetch(url);
  return handleResponse<FeedResponse>(res);
}

export async function fetchExplanation(
  recommendationId: string
): Promise<RecommendationExplanation> {
  const res = await fetch(`${BASE_URL}/recommendations/${recommendationId}/explanation`);
  return handleResponse<RecommendationExplanation>(res);
}

export async function fetchModels(): Promise<ModelStatus[]> {
  const res = await fetch(`${BASE_URL}/models`);
  return handleResponse<ModelStatus[]>(res);
}

export async function fetchEvaluations(): Promise<EvaluationResult[]> {
  const res = await fetch(`${BASE_URL}/evaluations`);
  return handleResponse<EvaluationResult[]>(res);
}

export async function recordInteraction(
  userId: string,
  articleId: string,
  interactionType: string
): Promise<{ status: string }> {
  const res = await fetch(`${BASE_URL}/interactions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_id: userId,
      article_id: articleId,
      interaction_type: interactionType,
    }),
  });
  return handleResponse<{ status: string }>(res);
}

// --- Pass 2 API endpoints ---
export async function classifyText(title: string, abstract: string = '', modelType: string = 'baseline') {
  const res = await fetch(`${BASE_URL}/nlp/classify`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title, abstract, model_type: modelType }),
  });
  return handleResponse<any>(res);
}

export async function extractEntities(text: string, articleId?: string) {
  const res = await fetch(`${BASE_URL}/nlp/ner/extract`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, article_id: articleId }),
  });
  return handleResponse<any>(res);
}

export async function fetchAblations(maxImpressions: number = 80) {
  const res = await fetch(`${BASE_URL}/ablations?max_impressions=${maxImpressions}`);
  return handleResponse<any>(res);
}

export async function computeSampleSize(baselineCtr: number = 0.04, mdeRelative: number = 0.05, alpha: number = 0.05, power: number = 0.80) {
  const res = await fetch(`${BASE_URL}/experiments/sample-size`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ baseline_ctr: baselineCtr, mde_relative: mdeRelative, alpha, power }),
  });
  return handleResponse<any>(res);
}

export async function runBootstrapTest(modelA: string = 'hybrid', modelB: string = 'ranker', numBootstraps: number = 200) {
  const res = await fetch(`${BASE_URL}/experiments/bootstrap-test`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_a: modelA, model_b: modelB, num_bootstraps: numBootstraps }),
  });
  return handleResponse<any>(res);
}

export async function createOnboardingPersona(selectedCategories: string[], seedArticleIds?: string[]) {
  const res = await fetch(`${BASE_URL}/cold-start/onboard`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ selected_categories: selectedCategories, seed_article_ids: seedArticleIds }),
  });
  return handleResponse<any>(res);
}

export async function fetchColdStartSegments() {
  const res = await fetch(`${BASE_URL}/cold-start/segments`);
  return handleResponse<any>(res);
}

export async function fetchMonitoringStats() {
  const res = await fetch(`${BASE_URL}/monitoring/stats`);
  return handleResponse<any>(res);
}

export async function fetchDriftReport() {
  const res = await fetch(`${BASE_URL}/monitoring/drift`);
  return handleResponse<any>(res);
}

export async function fetchAnnotationQueue(topN: number = 10, strategy: string = 'entropy') {
  const res = await fetch(`${BASE_URL}/annotations/queue?top_n=${topN}&strategy=${strategy}`);
  return handleResponse<any>(res);
}

export async function getMockLLMPreLabel(title: string, abstract: string = '') {
  const res = await fetch(`${BASE_URL}/annotations/pre-label`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title, abstract }),
  });
  return handleResponse<any>(res);
}

export async function submitAnnotation(articleId: string, trueLabel: string, annotatorId: string = 'annotator_1') {
  const res = await fetch(`${BASE_URL}/annotations/submit`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ article_id: articleId, true_label: trueLabel, annotator_id: annotatorId }),
  });
  return handleResponse<any>(res);
}

export async function computeAgreement(rater1: string[], rater2: string[]) {
  const res = await fetch(`${BASE_URL}/annotations/agreement`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ rater1_labels: rater1, rater2_labels: rater2 }),
  });
  return handleResponse<any>(res);
}

export async function fetchRegistryModels() {
  const res = await fetch(`${BASE_URL}/registry/models`);
  return handleResponse<any>(res);
}

export async function fetchSegmentDiagnostics() {
  const res = await fetch(`${BASE_URL}/diagnostics/segments`);
  return handleResponse<any>(res);
}

export async function fetchFailureArchetypes() {
  const res = await fetch(`${BASE_URL}/diagnostics/failures`);
  return handleResponse<any>(res);
}

