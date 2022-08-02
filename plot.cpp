#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TH1F.h>
#include <vector>

void plot()
{

    TFile *f = TFile::Open("regressed_target.root", "READ");
    TTree *t; f->GetObject("tree", t);

    std::vector<float> *test = 0;

}

int main()
{
    plot();
}