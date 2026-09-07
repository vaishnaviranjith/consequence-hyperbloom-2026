export type EntityType =
  | 'service'
  | 'place'
  | 'capability'
  | 'process'
  | 'resource'
  | 'constraint'
  | 'transport'
  | 'metric'

export type RelationshipType =
  | 'located_at'
  | 'constrained_by'
  | 'affects'
  | 'requires'
  | 'blocks'
  | 'may_depend_on'

export type PropagationOperator =
  | 'location'
  | 'capacity'
  | 'access'
  | 'emergency'
  | 'distance'
  | 'resource'
  | 'schedule'

export interface Scenario {
  id: string
  title: string
  domain: string
  evidenceIds: string[]
  changeProposalId: string
}

export interface Evidence {
  id: string
  title: string
  type: 'PLAN' | 'POLICY' | 'SCHEDULE' | 'MEMO' | 'NOTE' | 'PDF' | 'IMAGE' | 'TEXT'
  summary: string
  refs: string[]
  chunkIds?: string[]
}

export interface EvidenceChunk {
  id: string
  evidenceId: string
  content: string
  locator?: string
}

export interface Entity {
  id: string
  type: EntityType
  label: string
  attributes: Record<string, unknown>
  sourceRefs: string[]
  sourceChunkIds?: string[]
}

export interface Relationship {
  id: string
  from: string
  to: string
  type: RelationshipType
  operator: PropagationOperator
  strength: number
  sourceRefs: string[]
  sourceChunkIds?: string[]
  activeIn?: Array<'baseline' | 'counterfactual'>
}

export interface ChangeProposal {
  id: string
  statement: string
  from: string
  to: string
  kind: 'relocation'
}

export interface WorldState {
  location: string
  values: Record<string, unknown>
  entities: Entity[]
  relationships: Relationship[]
}

export interface CounterfactualState {
  before: WorldState
  after: WorldState
  delta: {
    location: { before: string; after: string }
    changedAttributes: string[]
  }
}

export interface RiskScore {
  severity: number
  confidence: number
  impactScore: number
  decisionPriority: number
}

export interface MitigationAction {
  title: string
  owner: string
  verification: string
}

export interface Consequence extends RiskScore {
  id: string
  title: string
  classification: 'direct' | 'indirect'
  depth: number
  explanation: string
  evidenceRefs: string[]
  evidenceChunkRefs?: string[]
  path: string[]
  relationshipTypes: RelationshipType[]
  mitigation: MitigationAction
  delta: string
  contradictions: number
}

export interface ExtractionEntity {
  name: string
  type: EntityType
  description: string
  attributes: Record<string, unknown>
  confidence: number
  evidence_chunk_ids: string[]
}

export interface ExtractionRelationship {
  source: string
  target: string
  relationship_type: RelationshipType
  operator: PropagationOperator
  strength: number
  confidence: number
  rationale: string
  evidence_chunk_ids: string[]
}

export interface ExtractionUncertainty {
  claim: string
  confidence: number
  evidence_chunk_ids: string[]
}

export interface ExtractionPayload {
  entities: ExtractionEntity[]
  relationships: ExtractionRelationship[]
  uncertainties: ExtractionUncertainty[]
}

export interface DependencyHypothesis {
  id: string
  sourceEntityId: string
  targetEntityId: string
  relationshipType: 'may_depend_on'
  status: 'HYPOTHESIS'
  confidence: number
  rationale: string
  evidenceChunkIds: string[]
  verificationTest: string
}

export type MutationOperation =
  | { type: 'REMOVE_EDGE'; source: string; target: string; relationshipType: RelationshipType }
  | { type: 'ADD_EDGE'; source: string; target: string; relationshipType: RelationshipType; operator: PropagationOperator }
  | { type: 'UPDATE_NODE'; nodeId: string; attributes: Record<string, unknown> }

export interface MutationProposal {
  statement: string
  confidence: number
  rationale: string
  operations: MutationOperation[]
}

export interface AIProviderResult<T> {
  status: 'ok' | 'not_configured' | 'rejected'
  data?: T
  errors?: string[]
}

export interface AIReasoningProvider {
  extractEvidence(chunks: EvidenceChunk[]): Promise<AIProviderResult<EvidenceChunk[]>>
  extractEntitiesAndRelationships(chunks: EvidenceChunk[]): Promise<AIProviderResult<ExtractionPayload>>
  proposeHiddenDependencies(chunks: EvidenceChunk[]): Promise<AIProviderResult<DependencyHypothesis[]>>
  interpretChange(text: string, world: WorldState): Promise<AIProviderResult<MutationProposal>>
  generateExplanation(consequence: Consequence): Promise<AIProviderResult<string>>
  generateMitigations(consequence: Consequence): Promise<AIProviderResult<MitigationAction[]>>
}
