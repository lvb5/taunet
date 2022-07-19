"""
"""
#? What are these? 
FEATURES = [
    'TauJetsAuxDyn.mu', # average of two following effects:
    'TauJetsAuxDyn.nVtxPU', #number of vertices (tracking effects), Terry doesn't use
    'TauJetsAuxDyn.rho', #ambient transverse energy density (pileup), Terry doesn't use
    'TauJetsAuxDyn.ClustersMeanCenterLambda',
    'TauJetsAuxDyn.ClustersMeanFirstEngDens',
    'TauJetsAuxDyn.ClustersMeanSecondLambda',
    'TauJetsAuxDyn.ClustersMeanPresamplerFrac',
    'TauJetsAuxDyn.ClustersMeanEMProbability',
    'TauJetsAuxDyn.ptIntermediateAxisEM/TauJetsAuxDyn.ptIntermediateAxis', # Terry doesn't use
    'TauJetsAuxDyn.ptCombined',
    'TauJetsAuxDyn.ptPanTauCellBased/TauJetsAuxDyn.ptCombined',
    'TauJetsAuxDyn.ptIntermediateAxis/TauJetsAuxDyn.ptCombined',
    'TauJetsAuxDyn.etaPanTauCellBased',
    # 'TauJetsAuxDyn.PanTau_BDTValue_1p0n_vs_1p1n',
    # 'TauJetsAuxDyn.PanTau_BDTValue_1p1n_vs_1pXn',
    # 'TauJetsAuxDyn.PanTau_BDTValue_3p0n_vs_3pXn',
    # 'TauJetsAuxDyn.nTracks',
    'TauJetsAuxDyn.PFOEngRelDiff', #upsilon; quantifies polar
    'TauJetsAuxDyn.ptTauEnergyScale',
    'TauJetsAuxDyn.NNDecayMode',
    'TauJetsAuxDyn.NNDecayModeProb_1p0n',
    'TauJetsAuxDyn.NNDecayModeProb_1p1n',
    'TauJetsAuxDyn.NNDecayModeProb_1pXn',
    'TauJetsAuxDyn.NNDecayModeProb_3p0n',
    'TauJetsAuxDyn.NNDecayModeProb_3pXn',
    ]

TRUTH_FIELDS = [
    # 'TauJetsAuxDyn.truthPtVis'
    'TauJetsAuxDyn.truthPtVisDressed',
    'TauJetsAuxDyn.truthEtaVisDressed',
    # 'TauJetsAuxDyn.truthPhiVisDressed',
    ]

OTHER_TES = [
    'TauJetsAuxDyn.ptCombined',
    'TauJetsAuxDyn.ptFinalCalib',
]

TARGET_FIELD = 'TauJetsAuxDyn.truthPtVisDressed/TauJetsAuxDyn.ptCombined'
