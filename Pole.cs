using System;
using Fiourp;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Input;

namespace CartPole;

public class Pole : Entity
{
    private Cart cart;

    private Vector2 Prev;
    public Pole(Cart cart) : base(ConstrainedInRadius(cart.MiddlePos, new Vector2(Rand.NextInt(-20, 20), -100), 200), 200, 10, Sprite.None)
    {
        this.cart = cart;
    }

    private static Vector2 ConstrainedInRadius(Vector2 center, Vector2 pos, float radius)
        => (pos - center).Normalized() * radius + center;

    public override void Update()
    {
        base.Update();

        Prev = Pos;
        Pos = ConstrainedInRadius(cart.MiddlePos, Pos + Vector2.UnitY * 10, 200);

        if (Input.GetKeyDown(Keys.Z))
            Pos = cart.MiddlePos - Vector2.UnitY * 200;
    }

    public override void Render()
    {
        base.Render();
        
        Drawing.DrawLine(cart.MiddlePos, Pos, Color.LightBlue, 10);
    }

    public float GetAngle(out float AngularVelocity)
    {
        float angle = VectorHelper.GetAngle(-Vector2.UnitY, Pos - cart.MiddlePos);
        AngularVelocity = Math.Abs(angle - VectorHelper.GetAngle(Vector2.UnitX, Prev - cart.MiddlePos));
        return angle;
    }

    public void Reset()
    {
        //Pos = ConstrainedInRadius(cart.MiddlePos, cart.MiddlePos - Vector2.UnitY * 200 + Rand.NextInt(0, 0) * Vector2.UnitX, 200);
        //Pos = ConstrainedInRadius(cart.MiddlePos, cart.MiddlePos - Vector2.UnitY * 200 + Rand.NextInt(-15, 15) * Vector2.UnitX, 200);
        Pos = ConstrainedInRadius(cart.MiddlePos, cart.MiddlePos - Vector2.UnitY * 200 + Rand.NextInt(-5, 5) * Vector2.UnitX, 200);
    }
}