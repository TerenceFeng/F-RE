#pragma once

enum BxDFType { BSDF_NONE };

// class BSDF
// {
//    public:
//     Color f(const Vector &woW, const Vector &wiW, const BxDFType flags);
//     Color Sample_f(const Vector &woW, Vector &wiW, const BSDFSample &bsdfSample,
//                    float &pdf, const BxDFType flags, BxDFType &sampledType);
//     float Pdf(I wo : Vector, I wi : Vector);
// };

class BxDF
{
   public:
    virtual ~BxDF()
    {
    }
    BxDF(BxDFType t) : type(t)
    {
    }
    bool MatchesFlags(BxDFType flags) const
    {
        return (type & flags) == type;
    }

    virtual Color f(const Vector &wo, const Vector &wi) const = 0;

    // A usefule generalization: cosine weighted
    virtual Color Sample_f(const Vector &wo, Vector *wi,
            float u1, float u2, float *pdf) const {
        // Cosine-sample the hemisphere, flipping the direction if necessary
        *wi = CosineSampleHemisphere(u1, u2);
        if (wo.z < 0.) wi->z *= -1.f;
        *pdf = Pdf(wo, *wi);
        return f(wo, *wi);
    }
    virtual float Pdf(const Vector &wo, const Vector &wi) const {
        return SameHemisphere(wo, wi) ? AbsCosTheta(wi) * INV_PI : 0.f;
    }

   public:
    const BxDFType type;
};

class SpecularReflection : public BxDF
{
   public:
    virtual Color f(const Vector &, const Vector &) const
    {
        return {0.0f, 0.0f, 0.0f};
    }

    // wo: to observer
    // wi: incident light
    // u1,u2: sample position
    // pdf: ratio of current sample in total BxDF distribution
    virtual Color Sample_f(const Vector &wo, Vector *wi, float u1, float u2,
                           float *pdf) const
    {
        // Compute perfect specular reflection direction
        *wi = Vector(-wo.x, -wo.y, wo.z);
        *pdf = 1.f;
        return {0.0f, 0.0f, 0.0f};
        // return fresnel->Evaluate(CosTheta(wo)) * R / AbsCosTheta(*wi);
    }
    virtual float Pdf(const Vector &wo, const Vector &wi) const
    {
        return 0.;
    }

   private:
    Color R;
    // Fresnel *fresnel;
};

// Lambertian
class DiffuseReflection : public BxDF
{
   public:
    DiffuseReflection(const Color &_R) : R(_R), BxDF(BSDF_NONE)
    {
    }
    virtual Color Sample_f(const Vector &wo, Vector *wi, float u1, float u2,
                           float *pdf) const
    {
        // Compute perfect specular reflection direction
        *wi = Vector(-wo.x, -wo.y, wo.z);
        *pdf = 1.f;
        return {0.0f, 0.0f, 0.0f};
        // return fresnel->Evaluate(CosTheta(wo)) * R / AbsCosTheta(*wi);
    }
    virtual Color f(const Vector &wo, const Vector &wi) const
    {
        return R * 1.0f / 3.14159f /* INV_PI */;
    }
    virtual float Pdf(const Vector &wo, const Vector &wi) const
    {
        return 0.;
    }

   private:
    Color R;
};
