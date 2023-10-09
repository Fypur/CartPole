using System;
using Fiourp;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Input;

namespace CartPole;

public class Cart : Entity
{
    public float Velocity;
    public Cart() : base(Engine.ScreenSize / 2, 70, 30, new Sprite(Color.White))
    {
    }

    public override void Update()
    {
        base.Update();

        if (Input.GetKey(Keys.D))
            Velocity += 0.5f;
        else if (Input.GetKey(Keys.Q))
            Velocity -= 0.5f;

        Pos.X += Velocity;
        
        Debug.LogUpdate(Velocity);
    }

    public void Update(int action)
    {
        if (action == 0)
            Velocity -= 0.5f;
        else if (action == 1)
            Velocity += 0.5f;
        else
            throw new Exception("Action number is bigger than expected");

        Pos.X += Velocity;
    }

    public void Reset()
    {
        Pos.X = Engine.ScreenSize.X / 2;
        Velocity = 0;
    }
}