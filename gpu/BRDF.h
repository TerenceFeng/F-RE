#pragma once

class BxDF
{
   public:
    virtual Color f(const Normal &nr, const Vector &wi,
                    const Vector &wo) const = 0;
};

class SpecularReflection : public BxDF
{
   public:
    SpecularReflection(const Color &r) : R(r)
    {
    }
    virtual Color f(const Normal &nr, const Vector &wi, const Vector &wo) const
    {
        return {0.0f, 0.0f, 0.0f};
    }

   private:
    Color R;
};

// Lambertian
class DiffuseReflection : public BxDF
{
   public:
    DiffuseReflection(const Color &r) : R(r)
    {
    }
    virtual Color f(const Normal &nr, const Vector &wi, const Vector &wo) const
    {
        return Vector::Scale(R.v, 1.0f / 3.14159f);
        // return R * INV_PI;
    }

   private:
    Color R;
};
