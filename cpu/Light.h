
/* ====================================================
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : Light.h
# ====================================================*/

#ifndef _LIGHT_H
#define _LIGHT_H

// #include "World.h"
#include "RGBColor.h"
#include "Sampler.h"
#include "ShadeRec.h"
#include "Material.h"
#include "Utilities.h"
#include "object/Object.h"
#include <vector>
#include <cfloat>

bool in_shadow(const Ray&);

class Light
{
public:
	Light(void) {}
	virtual ~Light(void) {}

	virtual Vector3D get_direction(ShadeRec& sr) = 0;
	virtual RGBColor L(ShadeRec& sr) = 0;
	virtual float G(const ShadeRec&) const
    {
        return 1;
    }
	virtual float pdf(ShadeRec&) const
    {
        return 1;
    }
	virtual void set_sampler(Sampler *s_)
    {
        sampler_ptr = s_;
    }

protected:
	Sampler *sampler_ptr;
};

class Ambient: public Light
{
public:
	Ambient(void):
        Light(),
        ls(1.0f),
        color(1.0f)
    {}
    Ambient(float ls_, RGBColor color_):
        Light(),
        ls(ls_),
        color(color_)
    {}
    virtual Vector3D get_direction(ShadeRec& sr) {
        return Vector3D(0.0);
    }
    virtual RGBColor L(ShadeRec& sr)
    {
        return color * ls;
    }
    inline void scale_radiance(const float b)
    {
        ls = b;
    }
    inline void set_color(const RGBColor& color_)
    {
        color = color_;
    }

private:
	float ls;
	RGBColor color;
};

class PointLight: public Light
{
public:
    PointLight():
        Light(),
        ls(1.0f),
        color(1.0f),
        location(Vector3D(0.0f))
    {}

    PointLight(float ls_, const RGBColor& color_, const Vector3D& location_):
        Light(),
        ls(ls_),
        color(color_),
        location(location_)
    {}

    virtual Vector3D get_direction(ShadeRec& sr)
    {
        return (location - sr.hit_point).hat();
    }
    virtual RGBColor L(ShadeRec& sr)
    {
        return color * ls;
    }

private:
	float ls;
	RGBColor color;
	Point3D location;
};

class AreaLight: public Light
{
public:
	AreaLight() {}
    AreaLight(Object* object_ptr_, Material* material_ptr_)
    {
        material_ptr = material_ptr_;
        object_ptr = object_ptr_;
    }

    ~AreaLight()
    {
        delete object_ptr;
    }

    virtual RGBColor L(ShadeRec& sr)
    {
        float ndotd = -light_normal * wi;
        if (ndotd > 0.0f)
            return material_ptr->get_Le(sr);
        else
            return BLACK;
    }
    virtual Vector3D get_direction(ShadeRec& sr)
    {
        sample_point = object_ptr->sample();
        light_normal = object_ptr->get_normal(sample_point);
        wi = sample_point - sr.hit_point;
        wi.normalize();
        return wi;
    }
    virtual float G(const ShadeRec& sr) const
    {
        float ndotd = -light_normal * wi;
        float dsqr = sample_point.distance_sqr(sr.hit_point);
        return (ndotd / dsqr);
    }
    virtual float pdf(ShadeRec& sr) const
    {
        return object_ptr->pdf(sr);
    }

    void set_object(Object* object_ptr_)
    {
        object_ptr = object_ptr_;
    }
    void set_material(Material* material_ptr_)
    {
        material_ptr = material_ptr_;
    }

private:
	bool V(const Ray&) const;
	Object* object_ptr;
	Material* material_ptr;
	Point3D sample_point;
	Normal light_normal;
	Vector3D wi;
};

class AmbientOccluder: public Light
{
public:
    AmbientOccluder(float ls_ = 1, const RGBColor& color_ = WHITE, const RGBColor& min_amount_ = WHITE):
        ls(ls_),
        color(color_),
        min_amount(min_amount_)
    {}
    virtual Vector3D get_direction(ShadeRec& sr)
    {
        Point3D sp = sampler_ptr->sample_unit_hemisphere();
        return (u * sp.x + v * sp.y + w * sp.z);
    }

    virtual RGBColor L(ShadeRec& sr)
    {
        w = sr.normal;
        v = w ^ Vector3D(0.0072, 1.0, 0.0034);
        v.normalize();
        u = v ^ w;
        Ray shadow_ray(sr.hit_point, get_direction(sr));
        if (in_shadow(shadow_ray))
            return min_amount * ls * color;
        else
            return color * ls;
    }

private:
	Vector3D u, v, w;
	RGBColor color;
	float ls;
	RGBColor min_amount;
};

class EnviormentLight: public Light
{
public:
	EnviormentLight():
        Light()
    {}

    EnviormentLight(Sampler *s_, Material *m_):
        Light(),
        material_ptr(m_)
    {
        sampler_ptr = s_;
    }

    void set_material(Material* material_ptr_)
    {
        material_ptr = material_ptr_;
    }
    virtual Vector3D get_direction(ShadeRec& sr)
    {
        w = sr.normal;
        v = Vector3D(0.0034, 1, 0.0071) ^ w;
        v.normalize();
        u = v ^ w;
        Point3D sp = sampler_ptr->sample_unit_hemisphere();
        return (u * sp.x + v * sp.y + w * sp.z);
    }	virtual RGBColor L(ShadeRec& sr)
    {
        return material_ptr->get_Le(sr);
    }

private:
    Material* material_ptr;
    Vector3D u, v, w;
    Vector3D wi;
};

#endif
